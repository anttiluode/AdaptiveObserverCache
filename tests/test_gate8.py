import unittest

try:
    import torch
    import transformers  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Gate 8 requires torch + transformers")
class Gate8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gate8_active_probing import build_active_probe_suite

        (
            cls.base,
            cls.tokenizer,
            cls.projections,
            cls.history,
        ) = build_active_probe_suite()

    def test_active_policy_never_buys_historically_useless_probe(self):
        from gate6_relational_alias import canonical_key
        from gate8_active_probing import (
            active_bind,
            current_probe_field,
            informative_probes,
        )

        for projection in self.projections:
            family = projection.case.family
            pair = (
                canonical_key(self.tokenizer, family.canonical0),
                canonical_key(self.tokenizer, family.canonical1),
            )
            field = current_probe_field(projection)
            allowed = set(informative_probes(projection))
            for key in pair:
                trace = active_bind(
                    key, pair, self.history, field, projection
                )
                self.assertTrue(set(trace.probes).issubset(allowed))
                self.assertLessEqual(trace.cost, 2)

    def test_random_order_has_higher_exact_expected_cost(self):
        from gate6_relational_alias import canonical_key
        from gate8_active_probing import (
            active_bind,
            current_probe_field,
            exact_random_order_expected_cost,
        )

        projection = self.projections[0]
        family = projection.case.family
        pair = (
            canonical_key(self.tokenizer, family.canonical0),
            canonical_key(self.tokenizer, family.canonical1),
        )
        field = current_probe_field(projection)
        key = pair[0]

        active = active_bind(
            key, pair, self.history, field, projection
        )
        random_cost = exact_random_order_expected_cost(
            key, self.history, field
        )
        self.assertLess(active.cost, random_cost)

    def test_gate8_receipt_passes_cost_and_history_attackers(self):
        from gate8_experiment import build_receipt

        receipt = build_receipt()
        self.assertTrue(receipt["pass"])
        summary = receipt["summary"]
        self.assertGreaterEqual(summary["active_accuracy"], 0.95)
        self.assertLessEqual(summary["mean_active_probe_cost"], 1.50)
        self.assertLessEqual(summary["max_active_probe_cost"], 2)
        self.assertGreaterEqual(
            summary["mean_random_order_expected_cost"], 2.80
        )
        self.assertLessEqual(
            summary["active_to_random_cost_ratio"], 0.55
        )
        self.assertLessEqual(summary["shuffled_history_accuracy"], 0.10)


if __name__ == "__main__":
    unittest.main()
