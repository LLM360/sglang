"""Tests for the --no-cache-thoughts CLI flag on ServerArgs."""

import argparse
import unittest

from sglang.srt.server_args import ServerArgs


class TestNoCacheThoughtsCliFlag(unittest.TestCase):
    def test_default_is_false(self):
        s = ServerArgs(model_path="dummy")
        self.assertFalse(s.no_cache_thoughts)

    def test_argparse_sets_to_true(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        ns = parser.parse_args(["--model-path", "dummy", "--no-cache-thoughts"])
        self.assertTrue(ns.no_cache_thoughts)


if __name__ == "__main__":
    unittest.main()
