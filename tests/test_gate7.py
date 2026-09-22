import unittest

try:
    import torch
    import transformers  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Gate 7 requires torch + transformers")
class Gate7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gate7_experiment import build_receipt

        cls.receipt = build_receipt()

    def test_active_identity_recovery_passes(self):
        self.assertTrue(self.receipt["pass"])
        summary = self.receipt["summary"]
        self.assertGreaterEqual(summary["active_binding_accuracy"], 0.95)
        self.assertGreaterEqual(
            summary["reversed_order_dual_success_rate"], 0.95
        )

    def test_active_beats_best_fixed_equal_budget_probe(self):
        summary = self.receipt["summary"]
        self.assertGreaterEqual(
            summary["active_advantage_over_best_fixed"], 0.10
        )
        self.assertEqual(
            self.receipt["identity_recovery"]["challenge_rounds_per_case"],
            1,
        )

    def test_history_and_passive_attackers_fail(self):
        summary = self.receipt["summary"]
        self.assertLessEqual(summary["passive_static_slot_accuracy"], 0.50)
        self.assertLessEqual(summary["shuffled_history_accuracy"], 0.10)

    def test_selector_is_not_one_global_probe(self):
        self.assertGreaterEqual(
            self.receipt["summary"]["distinct_active_probes_used"], 3
        )

    def test_all_caches_are_immutable(self):
        self.assertTrue(
            all(row["cache_unchanged"] for row in self.receipt["results"])
        )


if __name__ == "__main__":
    unittest.main()
