import gc
import weakref
from concurrent.futures import Future
from unittest.mock import Mock, call

import pytest
import torch

from sglang.srt.model_loader import weight_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="stage-a-test-cpu")


class _TrackedState(dict):
    pass


class _ImmediateExecutor:
    """Synchronous executor that returns real Futures for lifetime tests."""

    instances = []

    def __init__(self, max_workers):
        self.max_workers = max_workers
        self.submitted = 0
        self.future_refs = {}
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def submit(self, function, path):
        self.submitted += 1
        future = Future()
        try:
            future.set_result(function(path))
        except BaseException as error:
            future.set_exception(error)
        self.future_refs[path] = weakref.ref(future)
        return future


class _SuccessFirstSet(set):
    """Make pop() choose a success even when the set contains a failure."""

    def pop(self):
        for future in self:
            if not future.cancelled() and future.exception() is None:
                self.remove(future)
                return future
        return super().pop()


def test_load_pt_file_prefers_mmap(monkeypatch):
    state = {"weight": torch.tensor([1])}
    load = Mock(return_value=state)
    monkeypatch.setattr(weight_utils.torch, "load", load)

    assert weight_utils._load_pt_file("model.bin") is state
    load.assert_called_once_with(
        "model.bin", map_location="cpu", weights_only=True, mmap=True
    )


def test_load_pt_file_falls_back_for_legacy_tar(monkeypatch):
    mmap_error = RuntimeError(
        "mmap can only be used with files saved with the new zip format"
    )
    legacy_error = RuntimeError(
        "Cannot use weights_only=True with files saved in the legacy .tar format"
    )
    state = {"weight": torch.tensor([1])}
    load = Mock(side_effect=[mmap_error, legacy_error, state])
    monkeypatch.setattr(weight_utils.torch, "load", load)

    assert weight_utils._load_pt_file("legacy.bin") is state
    assert load.call_args_list == [
        call("legacy.bin", map_location="cpu", weights_only=True, mmap=True),
        call("legacy.bin", map_location="cpu", weights_only=True, mmap=False),
        call("legacy.bin", map_location="cpu", weights_only=False, mmap=False),
    ]


def test_load_pt_file_reads_real_legacy_tar(tmp_path):
    path = tmp_path / "legacy.bin"
    expected = torch.arange(4)
    torch.save({"weight": expected}, path, _use_new_zipfile_serialization=False)

    loaded = weight_utils._load_pt_file(str(path))

    assert torch.equal(loaded["weight"], expected)


def test_load_pt_file_does_not_hide_checkpoint_errors(monkeypatch):
    load_error = RuntimeError("PytorchStreamReader failed reading zip archive")
    load = Mock(side_effect=load_error)
    monkeypatch.setattr(weight_utils.torch, "load", load)

    with pytest.raises(RuntimeError, match="PytorchStreamReader"):
        weight_utils._load_pt_file("corrupt.bin")

    load.assert_called_once_with(
        "corrupt.bin", map_location="cpu", weights_only=True, mmap=True
    )


def test_multithread_pt_iterator_bounds_and_releases_results(monkeypatch):
    state_refs = {}

    def load(path):
        state = _TrackedState({path: torch.tensor([1])})
        state_refs[path] = weakref.ref(state)
        return state

    _ImmediateExecutor.instances.clear()
    monkeypatch.setattr(weight_utils, "_load_pt_file", load)
    monkeypatch.setattr(
        weight_utils.concurrent.futures,
        "ThreadPoolExecutor",
        _ImmediateExecutor,
    )

    paths = [f"shard-{index}.bin" for index in range(10)]
    iterator = weight_utils.multi_thread_pt_weights_iterator(paths, max_workers=3)
    first_name, first_tensor = next(iterator)
    executor = _ImmediateExecutor.instances[-1]

    # Only one fixed-size window is submitted before the consumer advances.
    assert executor.submitted == 3
    gc.collect()
    # The completed Future must not pin the state currently being consumed.
    assert executor.future_refs[first_name]() is None
    assert state_refs[first_name]() is not None

    second_name, second_tensor = next(iterator)
    del first_tensor
    gc.collect()
    # Advancing releases the previous state instead of retaining every shard.
    assert state_refs[first_name]() is None
    assert executor.submitted == 4

    del second_name, second_tensor
    iterator.close()


def test_multithread_pt_iterator_raises_completed_failure_before_success(
    monkeypatch,
):
    state_refs = {}

    def load(path):
        if path == "bad.bin":
            raise RuntimeError("failed shard")
        state = _TrackedState({path: torch.tensor([1])})
        state_refs[path] = weakref.ref(state)
        return state

    def success_first_wait(pending, return_when):
        assert return_when == weight_utils.concurrent.futures.FIRST_COMPLETED
        return _SuccessFirstSet(pending), set()

    _ImmediateExecutor.instances.clear()
    monkeypatch.setattr(weight_utils, "_load_pt_file", load)
    monkeypatch.setattr(
        weight_utils.concurrent.futures,
        "ThreadPoolExecutor",
        _ImmediateExecutor,
    )
    monkeypatch.setattr(
        weight_utils.concurrent.futures,
        "wait",
        success_first_wait,
    )

    paths = ["good-0.bin", "bad.bin", "good-1.bin", "not-submitted.bin"]
    iterator = weight_utils.multi_thread_pt_weights_iterator(paths, max_workers=3)
    try:
        with pytest.raises(RuntimeError, match="failed shard"):
            next(iterator)
    finally:
        iterator.close()

    # The failed initial window must not yield a success or submit more work.
    assert _ImmediateExecutor.instances[-1].submitted == 3
    gc.collect()
    assert all(state_ref() is None for state_ref in state_refs.values())


def test_multithread_pt_iterator_loads_every_shard(monkeypatch):
    monkeypatch.setattr(
        weight_utils,
        "_load_pt_file",
        lambda path: {path: torch.tensor([int(path)])},
    )

    loaded = dict(
        weight_utils.multi_thread_pt_weights_iterator(
            [str(index) for index in range(7)], max_workers=2
        )
    )

    assert set(loaded) == {str(index) for index in range(7)}
    assert {name: value.item() for name, value in loaded.items()} == {
        str(index): index for index in range(7)
    }
