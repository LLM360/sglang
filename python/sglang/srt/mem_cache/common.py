from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, List, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictParams
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import support_triton
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


_split_unsupported_warned = set()


def warn_split_unsupported_once(backend_name: str) -> None:
    """Emit a one-time warning when a prefix-cache backend that does not implement
    --no-cache-thoughts split insertion receives a split kwarg. The split is dropped
    and the backend falls back to its default cache_finished_req behavior, so the
    feature gracefully no-ops on that backend.
    """
    if backend_name in _split_unsupported_warned:
        return
    _split_unsupported_warned.add(backend_name)
    logger.warning(
        "%s does not implement --no-cache-thoughts split insertion; "
        "thoughts will be cached normally on this backend (no-op).",
        backend_name,
    )


def derive_position_offsets(
    extend_prefix_lens: List[int],
    cached_positions_per_req: List[Optional["torch.Tensor"]],
) -> Optional[List[int]]:
    """Given per-request cached RoPE positions, return a per-request offset to add
    to (seq_len - 1) at decode time so decode positions continue from where the
    non-contiguous prefill cache hit left off.

    The offset captures the gap in RoPE space caused by tokens that exist in the
    cached entry's position layout but not in the contiguous-token-count layout:
        offset[i] = max(cached_positions[i]) - (extend_prefix_lens[i] - 1)
    A request without cached positions contributes 0 (no offset).

    Returns None if every entry is None (i.e. no request in the batch needs an
    offset; the legacy clamp(seq_lens - 1) path applies).
    """
    if all(p is None for p in cached_positions_per_req):
        return None
    offsets: List[int] = []
    for prefix_len, positions in zip(extend_prefix_lens, cached_positions_per_req):
        if positions is None or len(positions) == 0:
            offsets.append(0)
        else:
            offsets.append(int(positions.max().item()) - (int(prefix_len) - 1))
    return offsets


def derive_extend_position_start(
    extend_prefix_lens: List[int],
    cached_positions_per_req: List[Optional["torch.Tensor"]],
) -> Optional[List[int]]:
    """Given per-request cached RoPE positions, return the starting RoPE position for
    each request's extend (prefill) tokens.

    Args:
        extend_prefix_lens: per-request count of cached prefix tokens.
        cached_positions_per_req: per-request tensor of cached original positions,
            or None if no positions are cached for that request.

    Returns:
        None if every entry is None (i.e. no request has non-contiguous cached
        positions, so the legacy contiguous-positions path applies). Otherwise a
        list of per-request integer starts: max(cached_positions) + 1 when cached
        positions exist, else extend_prefix_lens[i] (legacy).
    """
    if all(p is None for p in cached_positions_per_req):
        return None
    starts: List[int] = []
    for prefix_len, positions in zip(extend_prefix_lens, cached_positions_per_req):
        if positions is None or len(positions) == 0:
            starts.append(int(prefix_len))
        else:
            starts.append(int(positions.max().item()) + 1)
    return starts


@dataclasses.dataclass
class NoCacheThoughtsSplit:
    """Plan for caching a finished reasoning request without its thought span.

    Dropping the generated <think>...</think> tokens shifts every answer token's index
    in the cached sequence DOWN by len(thoughts), but its KV still physically sits in
    the slot decode gave it. Paged KV requires slot % page_size == index % page_size,
    so the answer's KV must be RELOCATED into the slots the thoughts are vacating (which
    are page-congruent to the answer's new indices) before it can be cached and safely
    extended later. cache_finished_req consumes this plan as:
      1. move_kv_cache(move_dst, move_src)  -> slide the answer's KV left
      2. insert virtual_token_ids/positions, values from virtual_kv_indices[:aligned]
      3. free virtual_kv_indices[aligned:]  -> one page-aligned cut for the dead tail
    """

    # input + post-</think> answer (thoughts removed); the cached key sequence.
    virtual_token_ids: List[int]
    # The request's FULL original contiguous slot span S[0:total_len]. After the move,
    # S[0:kept_len] holds input+answer and S[kept_len:] is the dead tail (stale thought
    # slots + the answer's page-unaligned remainder), freed in one page-aligned cut.
    virtual_kv_indices: torch.Tensor
    # Original RoPE positions of the kept tokens (gapped where thoughts were); the K
    # vectors already encode these, so positions are preserved while only slots move.
    virtual_positions: torch.Tensor
    # Relocation: move the answer's current slots (move_src = S[answer_start:total_len])
    # into the page-congruent destination (move_dst = S[input_len:kept_len]). Empty when
    # there is no thought span or no answer (nothing to relocate).
    move_src: torch.Tensor
    move_dst: torch.Tensor


def split_kv_for_no_cache_thoughts(
    origin_input_ids: List[int],
    output_ids: List[int],
    req_to_token_slot: torch.Tensor,
    answer_start_position: int,
    committed_len: int,
) -> NoCacheThoughtsSplit:
    """Compute the split-insertion tensors for a finished reasoning request.

    When --no-cache-thoughts is enabled and a request emits `</think>`, the tokens
    between input_end and the `</think>` boundary are thoughts that must NOT be
    registered in the cross-request radix cache; the post-`</think>` answer must
    be inserted with the original RoPE positions preserved.

    Args:
        origin_input_ids: input token ids (positions 0..len(input)-1).
        output_ids: generated token ids (positions len(input)..len(input)+len(output)-1).
        req_to_token_slot: 1D tensor of kv_indices, one per token in the request's
            sequence (length must be >= committed_len).
        answer_start_position: absolute position (in the input+output index space)
            of the first answer token, i.e. the token immediately after `</think>`.
        committed_len: number of tokens whose KV is actually committed
            (``req.kv_committed_len``). The final generated token can appear in
            output_ids while its KV slot is uncommitted (overlap scheduling), in which
            case its req_to_token entry is the zero/unwritten sentinel (reserved page 0).
            Walking past committed_len would move/free that sentinel and double-free
            page 0, so we cap here exactly as the normal cache_finished_req path does.

    Returns:
        NoCacheThoughtsSplit (see that dataclass for field semantics): the cached key
        sequence (input + answer), the request's committed slot span, the kept tokens'
        original RoPE positions, and the move_src/move_dst slot tensors that relocate
        the answer's KV into page-congruent slots.

    When answer_start_position >= committed_len, the answer hasn't been committed
    (e.g. the request was cut off mid-thought) and the result caches only the input
    prompt slice (no relocation).
    """
    input_len = len(origin_input_ids)
    # Cap at the committed KV length (see committed_len arg): never touch the
    # uncommitted trailing token's unwritten (page-0) slot.
    total_len = min(input_len + len(output_ids), committed_len)

    # Clamp answer_start so we behave sanely if reasoning never finished.
    answer_start = min(answer_start_position, total_len)

    think_len = answer_start - input_len  # decoded thought tokens to drop
    answer_count = max(total_len - answer_start, 0)
    answer_output_offset = answer_start - input_len  # index into output_ids
    kept_len = input_len + answer_count  # cached sequence length (= total_len - think_len)

    # The full input prompt (including any <think>\n priming tail from
    # add_generation_prompt) stays in the cached entry — TITO rollouts feed
    # turn N+1's input as raw token IDs that include turn N's prompt verbatim,
    # so keeping the priming preserves cache alignment in that flow. The answer slice
    # is bounded by answer_count so an uncommitted trailing token is never cached.
    virtual_token_ids = list(origin_input_ids) + list(
        output_ids[answer_output_offset : answer_output_offset + answer_count]
        if answer_count > 0
        else []
    )

    # Original RoPE positions of the kept tokens: input is contiguous, then the answer
    # keeps its ORIGINAL positions (a gap where the thoughts were). The K vectors already
    # encode these positions, so we must not renumber them; only the physical slots move.
    kept_positions = list(range(input_len)) + list(range(answer_start, total_len))

    device = req_to_token_slot.device
    slots = req_to_token_slot.to(torch.int64)

    # Hand cache_finished_req the request's FULL original slot span. It takes cached
    # values from slots[:page_aligned(kept_len)] (after the move) and frees
    # slots[page_aligned(kept_len):] in a single page-aligned cut — that one cut covers
    # both the stale thought slots and the answer's unaligned tail with no boundary
    # double-free.
    virtual_kv_indices = slots[:total_len].clone()

    # Relocate the answer's KV LEFT by think_len slots so each answer token lands on the
    # slot that originally held its NEW index (slot % page_size == index % page_size).
    # The destination slots[input_len:kept_len] are exactly the thought slots plus the
    # answer's leading slots — all owned privately by this finished request, so the move
    # cannot disturb the shared input prefix already in the radix. Skip when there is no
    # thought span (already aligned) or no answer (nothing to relocate).
    if think_len > 0 and answer_count > 0:
        move_src = slots[answer_start:total_len].clone()
        move_dst = slots[input_len:kept_len].clone()
    else:
        move_src = torch.empty((0,), dtype=torch.int64, device=device)
        move_dst = torch.empty((0,), dtype=torch.int64, device=device)

    virtual_positions = torch.tensor(kept_positions, dtype=torch.int64, device=device)

    return NoCacheThoughtsSplit(
        virtual_token_ids=virtual_token_ids,
        virtual_kv_indices=virtual_kv_indices,
        virtual_positions=virtual_positions,
        move_src=move_src,
        move_dst=move_dst,
    )

# Needs 2 + 1 slots for mamba request with prefix cache. 2 for ping pong cache, 1 for running mamba state.
MAMBA_STATE_PER_REQ_PREFIX_CACHE = 3
MAMBA_STATE_PER_REQ_NO_CACHE = 1

logger = logging.getLogger(__name__)


@triton.jit
def write_req_to_token_pool_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices,
    prefix_tensors,
    pre_lens,
    seq_lens,
    extend_lens,
    out_cache_loc,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0)

    req_pool_index = tl.load(req_pool_indices + pid)
    pre_len = tl.load(pre_lens + pid)
    seq_len = tl.load(seq_lens + pid)
    prefix_tensor = tl.load(prefix_tensors + pid).to(tl.pointer_type(tl.int64))

    # write prefix
    num_loop = tl.cdiv(pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < pre_len
        value = tl.load(prefix_tensor + offset, mask=mask)
        tl.store(
            req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + offset,
            value,
            mask=mask,
        )

    # NOTE: This can be slow for large bs
    cumsum_start = tl.cast(0, tl.int64)
    for i in range(pid):
        cumsum_start += tl.load(extend_lens + i)

    num_loop = tl.cdiv(seq_len - pre_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < (seq_len - pre_len)
        value = tl.load(out_cache_loc + cumsum_start + offset, mask=mask)
        tl.store(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + offset
            + pre_len,
            value,
            mask=mask,
        )


def write_cache_indices(
    out_cache_loc: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    req_pool_indices_cpu: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens_tensor: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    extend_lens_tensor: torch.Tensor,
    extend_lens_cpu: torch.Tensor,
    prefix_tensors: list[torch.Tensor],
    req_to_token_pool: ReqToTokenPool,
):
    if support_triton(get_global_server_args().attention_backend):
        prefix_pointers = torch.tensor(
            [t.data_ptr() for t in prefix_tensors],
            device=req_to_token_pool.device,
            dtype=torch.uint64,
        )
        # TODO: some tensors can be reused for ForwardBatchInfo (e.g., extend_lens, cumsum_start)
        write_req_to_token_pool_triton[(req_pool_indices_tensor.shape[0],)](
            req_to_token_pool.req_to_token,
            req_pool_indices_tensor,
            prefix_pointers,
            prefix_lens_tensor,
            seq_lens_tensor,
            extend_lens_tensor,
            out_cache_loc,
            req_to_token_pool.req_to_token.shape[1],
        )
    else:
        pt = 0
        for i in range(req_pool_indices_cpu.shape[0]):
            req_idx = req_pool_indices_cpu[i].item()
            prefix_len = prefix_lens_cpu[i].item()
            seq_len = seq_lens_cpu[i].item()
            extend_len = extend_lens_cpu[i].item()

            req_to_token_pool.write(
                (req_idx, slice(0, prefix_len)),
                prefix_tensors[i],
            )
            req_to_token_pool.write(
                (req_idx, slice(prefix_len, seq_len)),
                out_cache_loc[pt : pt + extend_len],
            )
            pt += extend_len


def get_last_loc(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    if (
        get_global_server_args().attention_backend != "ascend"
        and get_global_server_args().attention_backend != "torch_native"
    ):
        impl = get_last_loc_triton
    else:
        impl = get_last_loc_torch

    return impl(req_to_token, req_pool_indices_tensor, prefix_lens_tensor)


def get_last_loc_torch(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    return torch.where(
        prefix_lens_tensor > 0,
        req_to_token[req_pool_indices_tensor, prefix_lens_tensor - 1],
        torch.full_like(prefix_lens_tensor, -1),
    )


@triton.jit
def get_last_loc_kernel(
    req_to_token,
    req_pool_indices_tensor,
    prefix_lens_tensor,
    result,
    num_tokens,
    req_to_token_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    mask = offset < num_tokens

    prefix_lens = tl.load(prefix_lens_tensor + offset, mask=mask, other=0)
    req_pool_indices = tl.load(req_pool_indices_tensor + offset, mask=mask, other=0)

    token_mask = prefix_lens > 0
    token_index = req_pool_indices * req_to_token_stride + (prefix_lens - 1)
    tokens = tl.load(req_to_token + token_index, mask=token_mask, other=-1)

    tl.store(result + offset, tokens, mask=mask)


def get_last_loc_triton(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    BLOCK_SIZE = 256
    num_tokens = prefix_lens_tensor.shape[0]
    result = torch.empty_like(prefix_lens_tensor)
    grid = (triton.cdiv(num_tokens, BLOCK_SIZE),)

    get_last_loc_kernel[grid](
        req_to_token,
        req_pool_indices_tensor,
        prefix_lens_tensor,
        result,
        num_tokens,
        req_to_token.stride(0),
        BLOCK_SIZE,
    )
    return result


def alloc_token_slots(
    tree_cache: BasePrefixCache,
    num_tokens: int,
    backup_state: bool = False,
):
    allocator = tree_cache.token_to_kv_pool_allocator
    evict_from_tree_cache(tree_cache, num_tokens)

    state = None
    if backup_state:
        state = allocator.backup_state()

    out_cache_loc = allocator.alloc(num_tokens)

    if out_cache_loc is None:
        error_msg = (
            f"Out of memory. Try to lower your batch size.\n"
            f"Try to allocate {num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return (out_cache_loc, state) if backup_state else out_cache_loc


def evict_from_tree_cache(tree_cache: BasePrefixCache | None, num_tokens: int):
    if tree_cache is None:
        return

    if tree_cache.is_chunk_cache():
        return

    allocator = tree_cache.token_to_kv_pool_allocator

    if isinstance(allocator, SWATokenToKVPoolAllocator):
        # Hybrid allocator
        full_available_size = allocator.full_available_size()
        swa_available_size = allocator.swa_available_size()

        if full_available_size < num_tokens or swa_available_size < num_tokens:
            full_num_tokens = max(0, num_tokens - full_available_size)
            swa_num_tokens = max(0, num_tokens - swa_available_size)
            tree_cache.evict(
                EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
            )
    else:
        # Standard allocator
        if allocator.available_size() < num_tokens:
            tree_cache.evict(EvictParams(num_tokens=num_tokens))


def alloc_paged_token_slots_extend(
    tree_cache: BasePrefixCache,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
    backup_state: bool = False,
):
    # Over estimate the number of tokens: assume each request needs a new page.
    allocator = tree_cache.token_to_kv_pool_allocator
    num_tokens = extend_num_tokens + len(seq_lens_cpu) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens)

    state = None
    if backup_state:
        state = allocator.backup_state()

    out_cache_loc = allocator.alloc_extend(
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens,
    )

    if out_cache_loc is None:
        error_msg = (
            f"Prefill out of memory. Try to lower your batch size.\n"
            f"Try to allocate {extend_num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return (out_cache_loc, state) if backup_state else out_cache_loc


def alloc_req_slots(
    req_to_token_pool: ReqToTokenPool,
    reqs: list[Req],
    tree_cache: BasePrefixCache | None,
) -> list[int]:
    """Allocate request slots from the pool."""
    num_reqs = len(reqs)
    if isinstance(req_to_token_pool, HybridReqToTokenPool):
        mamba_available_size = req_to_token_pool.mamba_pool.available_size()
        factor = (
            MAMBA_STATE_PER_REQ_PREFIX_CACHE
            if tree_cache.supports_mamba()
            else MAMBA_STATE_PER_REQ_NO_CACHE
        )
        mamba_state_needed = num_reqs * factor
        if mamba_available_size < mamba_state_needed:
            if tree_cache is not None and tree_cache.supports_mamba():
                mamba_num = max(0, mamba_state_needed - mamba_available_size)
                tree_cache.evict(EvictParams(num_tokens=0, mamba_num=mamba_num))
    req_pool_indices = req_to_token_pool.alloc(reqs)

    if req_pool_indices is None:
        raise RuntimeError(
            "alloc_req_slots runs out of memory. "
            "Please set a smaller number for `--max-running-requests`. "
            f"{req_to_token_pool.available_size()=}, "
            f"{num_reqs=}, "
        )
    return req_pool_indices


def alloc_for_extend(
    batch: ScheduleBatch,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """
    Allocate KV cache for extend batch and write to req_to_token_pool.

    Returns:
        out_cache_loc: allocated cache locations
        req_pool_indices_device: request pool indices at a device tensor
        req_pool_indices: request pool indices as list
    """
    # free out-of-window swa tokens
    batch.maybe_evict_swa()

    prefix_tensors = [r.prefix_indices for r in batch.reqs]

    # Create tensors for allocation
    prefix_lens_cpu = torch.tensor(batch.prefix_lens, dtype=torch.int64)
    extend_lens_cpu = torch.tensor(batch.extend_lens, dtype=torch.int64)
    prefix_lens_device = prefix_lens_cpu.to(batch.device, non_blocking=True)
    extend_lens_device = extend_lens_cpu.to(batch.device, non_blocking=True)

    # Allocate req slots
    req_pool_indices = alloc_req_slots(
        batch.req_to_token_pool, batch.reqs, batch.tree_cache
    )
    req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
    req_pool_indices_device = req_pool_indices_cpu.to(batch.device, non_blocking=True)

    # Allocate KV cache (throws exception on failure)
    if batch.tree_cache.page_size == 1:
        out_cache_loc = alloc_token_slots(batch.tree_cache, batch.extend_num_tokens)
    else:
        # Paged allocation - build last_loc
        last_loc = [
            (t[-1:] if len(t) > 0 else torch.tensor([-1], device=batch.device))
            for t in prefix_tensors
        ]
        out_cache_loc = alloc_paged_token_slots_extend(
            tree_cache=batch.tree_cache,
            prefix_lens=prefix_lens_device,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=batch.seq_lens,
            seq_lens_cpu=batch.seq_lens_cpu,
            last_loc=torch.cat(last_loc),
            extend_num_tokens=batch.extend_num_tokens,
        )

    # Write to req_to_token_pool
    write_cache_indices(
        out_cache_loc,
        req_pool_indices_device,
        req_pool_indices_cpu,
        prefix_lens_device,
        prefix_lens_cpu,
        batch.seq_lens,
        batch.seq_lens_cpu,
        extend_lens_device,
        extend_lens_cpu,
        prefix_tensors,
        batch.req_to_token_pool,
    )

    return out_cache_loc, req_pool_indices_device, req_pool_indices


def alloc_paged_token_slots_decode(
    tree_cache: BasePrefixCache,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    token_per_req: int = 1,
) -> torch.Tensor:
    """Allocate paged KV cache for decode batch."""
    allocator = tree_cache.token_to_kv_pool_allocator
    # Over estimate the number of tokens: assume each request needs a new page.
    num_tokens = len(seq_lens) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens)

    out_cache_loc = allocator.alloc_decode(seq_lens, seq_lens_cpu, last_loc)

    if out_cache_loc is None:
        error_msg = (
            f"Decode out of memory. Try to lower your batch size.\n"
            f"Try to allocate {len(seq_lens) * token_per_req} tokens.\n"
            f"{available_and_evictable_str(tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)

    return out_cache_loc


def alloc_for_decode(batch: ScheduleBatch, token_per_req: int) -> torch.Tensor:
    """
    Allocate KV cache for decode batch and write to req_to_token_pool.

    Returns:
        out_cache_loc: allocated cache locations
    """

    batch.maybe_evict_swa()

    bs = batch.seq_lens.shape[0]

    if batch.tree_cache.page_size == 1:
        # Non-paged allocation
        out_cache_loc = alloc_token_slots(batch.tree_cache, bs * token_per_req)
    else:
        # Paged allocation
        last_loc = batch.req_to_token_pool.req_to_token[
            batch.req_pool_indices, batch.seq_lens - 1
        ]
        seq_lens_next = batch.seq_lens + token_per_req
        out_cache_loc = alloc_paged_token_slots_decode(
            tree_cache=batch.tree_cache,
            seq_lens=seq_lens_next,
            seq_lens_cpu=batch.seq_lens_cpu + token_per_req,
            last_loc=last_loc,
            token_per_req=token_per_req,
        )

    # Write to req_to_token_pool
    if batch.model_config.is_encoder_decoder:
        locs = batch.encoder_lens + batch.seq_lens
    else:
        locs = batch.seq_lens.clone()

    batch.req_to_token_pool.write(
        (batch.req_pool_indices, locs), out_cache_loc.to(torch.int32)
    )

    return out_cache_loc


def release_kv_cache(req: Req, tree_cache: BasePrefixCache, is_insert: bool = True):
    # MambaRadixCache may alloc mamba state before alloc KV cache
    if req.req_pool_idx is None:
        assert (
            tree_cache.supports_mamba()
        ), "Only MambaRadixCache allow freeing before alloc"
        # TODO (csy, hanming): clean up this early allocation logic
        if req.mamba_pool_idx is not None:
            tree_cache.req_to_token_pool.mamba_pool.free(
                req.mamba_pool_idx.unsqueeze(-1)
            )
            req.mamba_pool_idx = None
        return

    server_args = get_global_server_args()
    if (
        is_insert
        and getattr(server_args, "no_cache_thoughts", False)
        and getattr(req, "require_reasoning", False)
        and getattr(req, "answer_start_position", None) is not None
    ):
        # Skip the thought tokens from the shared prefix cache; insert only the
        # input + post-</think> answer slice, preserving original RoPE positions
        # for the answer (the input prompt keeps its contiguous positions).
        req_to_token_slot = tree_cache.req_to_token_pool.req_to_token[req.req_pool_idx]
        split = split_kv_for_no_cache_thoughts(
            origin_input_ids=req.origin_input_ids,
            output_ids=req.output_ids,
            req_to_token_slot=req_to_token_slot,
            answer_start_position=req.answer_start_position,
            committed_len=req.kv_committed_len,
        )
        tree_cache.cache_finished_req(req, is_insert=is_insert, split=split)
    else:
        tree_cache.cache_finished_req(req, is_insert=is_insert)

    # FIXME: SessionAwareCache.cache_finished_req sets req_pool_idx = None to
    # transfer KV ownership to the SessionSlot, so we skip the remaining
    # cleanup (overalloc free + pool slot free). This means over-allocated
    # tokens from speculative decoding are NOT freed between turns.
    if req.req_pool_idx is None:
        return

    start_p, end_p = req.pop_overallocated_kv_cache()

    global_server_args = get_global_server_args()
    page_size = global_server_args.page_size
    spec_algo = global_server_args.speculative_algorithm

    if spec_algo is None:
        assert (
            start_p == end_p
        ), f"Unexpected overallocated KV cache, {req.kv_committed_len=}, {req.kv_allocated_len=}"

    if page_size > 1:
        start_p = ceil_align(start_p, page_size)

    if start_p < end_p:
        indices_to_free = tree_cache.req_to_token_pool.req_to_token[req.req_pool_idx][
            start_p:end_p
        ]
        tree_cache.token_to_kv_pool_allocator.free(indices_to_free)
    # If the prefix cache doesn't manage mamba states, we must free them here.
    if isinstance(tree_cache.req_to_token_pool, HybridReqToTokenPool) and (
        not tree_cache.supports_mamba()
    ):
        assert (
            req.mamba_pool_idx is not None
        ), "mamba state is freed while the tree cache does not manage mamba states"
        tree_cache.req_to_token_pool.free_mamba_cache(req)
    tree_cache.req_to_token_pool.free(req)


def available_and_evictable_str(tree_cache: BasePrefixCache) -> str:
    return tree_cache.available_and_evictable_str()
