import unittest

try:
    import torch
    import transformers  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Gate 4 requires torch + transformers")
class Gate4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gate4_experiment import build_receipt

        cls.receipt = build_receipt()

    def test_binding_repairs_order_swap(self):
        self.assertTrue(self.receipt["pass"])
        summary = self.receipt["summary"]
        self.assertGreaterEqual(summary["binder_accuracy"], 0.95)
        self.assertGreaterEqual(
            summary["swapped_dual_identity_success_rate"], 0.95
        )

    def test_position_only_attackers_fail(self):
        summary = self.receipt["summary"]
        self.assertLessEqual(
            summary["static_identity_to_slot_accuracy"], 0.50
        )
        self.assertLessEqual(summary["inverted_binder_accuracy"], 0.10)

    def test_binder_does_not_use_attention_geometry_or_trust(self):
        binder = self.receipt["binder"]
        self.assertFalse(binder["uses_attention_KV_geometry"])
        self.assertFalse(binder["uses_trust_label"])

    def test_all_caches_are_immutable(self):
        self.assertTrue(
            all(row["cache_unchanged"] for row in self.receipt["results"])
        )


if __name__ == "__main__":
    unittest.main()
