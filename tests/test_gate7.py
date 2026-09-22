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
        from gate7_history_continuity import build_history_suite

        (
            cls.base,
            cls.tokenizer,
            cls.projections,
            cls.history,
        ) = build_history_suite()

    def test_one_probe_is_ambiguous_but_full_history_binds(self):
        from gate6_relational_alias import canonical_key
        from gate7_history_continuity import (
            bind_from_history,
            observe_current_behavior,
        )

        projection = self.projections[0]
        family = projection.case.family
        key = canonical_key(self.tokenizer, family.canonical0)
        observations = observe_current_behavior(projection)

        with self.assertRaises(RuntimeError):
            bind_from_history(
                key, self.history, observations, probe_count=1
            )

        binding = bind_from_history(
            key, self.history, observations, probe_count=5
        )
        self.assertGreaterEqual(binding.margin, 1)

    def test_incomplete_relation_graph_cannot_reach_current_alias(self):
        from gate6_relational_alias import canonical_key
        from gate7_history_continuity import incomplete_relation_bind

        projection = self.projections[0]
        family = projection.case.family
        key = canonical_key(self.tokenizer, family.canonical0)
        self.assertIsNone(
            incomplete_relation_bind(
                self.tokenizer, projection, key
            )
        )

    def test_gate7_receipt_passes_history_attackers(self):
        from gate7_experiment import build_receipt

        receipt = build_receipt()
        self.assertTrue(receipt["pass"])
        summary = receipt["summary"]
        self.assertGreaterEqual(summary["history_bound_accuracy"], 0.95)
        self.assertGreaterEqual(summary["mean_target_mass"], 0.80)
        self.assertLessEqual(summary["static_slot_accuracy"], 0.50)
        self.assertLessEqual(summary["one_probe_accuracy"], 0.50)
        self.assertLessEqual(summary["reset_history_accuracy"], 0.50)
        self.assertLessEqual(summary["shuffled_history_accuracy"], 0.10)
        self.assertLessEqual(
            summary["incomplete_relation_graph_accuracy"], 0.10
        )


if __name__ == "__main__":
    unittest.main()
