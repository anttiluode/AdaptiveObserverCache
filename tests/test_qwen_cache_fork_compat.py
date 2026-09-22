import unittest

from qwen_cache_fork_compat import (
    cache_fork_mode,
    fork_dynamic_cache,
    layer_tensor_pairs,
)


class _Layer:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values


class _NewCache:
    def __init__(self):
        self.layers = [
            _Layer("k0", "v0"),
            _Layer("k1", "v1"),
        ]


class _NewDynamicCache:
    def __init__(self, *, ddp_cache_data):
        self.received = list(ddp_cache_data)


class _OldCache:
    def to_legacy_cache(self):
        return (("old-k", "old-v"),)


class _OldDynamicCache:
    @classmethod
    def from_legacy_cache(cls, payload):
        obj = cls()
        obj.received = payload
        return obj


class CacheForkCompatTests(unittest.TestCase):
    def test_new_layer_api_uses_ddp_cache_data(self):
        cache = _NewCache()
        self.assertEqual(
            layer_tensor_pairs(cache),
            [("k0", "v0"), ("k1", "v1")],
        )
        self.assertEqual(
            cache_fork_mode(cache, _NewDynamicCache),
            "layer-tensor-constructor",
        )
        branch, mode = fork_dynamic_cache(
            cache, _NewDynamicCache, version="5.test"
        )
        self.assertEqual(mode, "layer-tensor-constructor")
        self.assertEqual(
            branch.received,
            [("k0", "v0"), ("k1", "v1")],
        )

    def test_old_api_keeps_legacy_conversion(self):
        cache = _OldCache()
        self.assertEqual(
            cache_fork_mode(cache, _OldDynamicCache),
            "legacy-conversion",
        )
        branch, mode = fork_dynamic_cache(
            cache, _OldDynamicCache, version="4.test"
        )
        self.assertEqual(mode, "legacy-conversion")
        self.assertEqual(branch.received, (("old-k", "old-v"),))


if __name__ == "__main__":
    unittest.main()
