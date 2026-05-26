"""Signature-level test: every non-RadixCache prefix-cache backend's cache_finished_req
must accept the split kwarg (either explicitly or via **kwargs) so the --no-cache-thoughts
routing in release_kv_cache doesn't raise TypeError on these backends.
"""

import inspect
import unittest


def _accepts_split(cls) -> bool:
    sig = inspect.signature(cls.cache_finished_req)
    has_split = "split" in sig.parameters
    has_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    return has_split or has_kwargs


class TestOtherBackendsAcceptSplit(unittest.TestCase):
    def test_swa_radix_cache(self):
        from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache

        self.assertTrue(_accepts_split(SWARadixCache), "SWARadixCache rejects split kwarg")

    def test_mamba_radix_cache(self):
        from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache

        self.assertTrue(
            _accepts_split(MambaRadixCache), "MambaRadixCache rejects split kwarg"
        )

    def test_session_aware_cache(self):
        from sglang.srt.mem_cache.session_aware_cache import SessionAwareCache

        self.assertTrue(
            _accepts_split(SessionAwareCache), "SessionAwareCache rejects split kwarg"
        )

    def test_radix_cache_cpp(self):
        try:
            from sglang.srt.mem_cache.radix_cache_cpp import RadixCacheCpp
        except Exception as e:
            self.skipTest(f"RadixCacheCpp not importable in this env: {e}")
        self.assertTrue(
            _accepts_split(RadixCacheCpp), "RadixCacheCpp rejects split kwarg"
        )

    def test_lmc_radix_cache(self):
        try:
            from sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache import (
                LMCRadixCache,
            )
        except Exception as e:
            self.skipTest(f"LMCRadixCache not importable in this env: {e}")
        self.assertTrue(
            _accepts_split(LMCRadixCache), "LMCRadixCache rejects split kwarg"
        )


if __name__ == "__main__":
    unittest.main()
