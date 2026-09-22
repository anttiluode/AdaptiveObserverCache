import unittest

try:
    import torch
    import transformers  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Gate 5 requires torch + transformers")
class Gate5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gate5_experiment import build_receipt

        cls.receipt = build_receipt()

    def test_generic_keys_replace_identity_table(self):
        self.assertTrue(self.receipt["pass"])
        state = self.receipt["persistent_state"]
        self.assertFalse(state["hard_coded_identity_table"])
        self.assertGreaterEqual(
            self.receipt["summary"]["unique_persistent_keys"], 12
        )

    def test_generic_binding_survives_order_changes(self):
        summary = self.receipt["summary"]
        self.assertGreaterEqual(summary["generic_bound_accuracy"], 0.95)
        self.assertGreaterEqual(
            summary["reversed_order_dual_success_rate"], 0.95
        )

    def test_attackers_fail(self):
        summary = self.receipt["summary"]
        self.assertLessEqual(
            summary["static_calibration_slot_accuracy"], 0.50
        )
        self.assertLessEqual(
            summary["wrong_persistent_key_accuracy"], 0.10
        )

    def test_all_caches_are_immutable(self):
        self.assertTrue(
            all(row["cache_unchanged"] for row in self.receipt["results"])
        )


if __name__ == "__main__":
    unittest.main()
