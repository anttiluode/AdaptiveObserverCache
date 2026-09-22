import unittest

try:
    import torch
    import transformers  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Gate 6 requires torch + transformers")
class Gate6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gate6_experiment import build_receipt

        cls.receipt = build_receipt()

    def test_multihop_alias_binding_passes(self):
        self.assertTrue(self.receipt["pass"])
        summary = self.receipt["summary"]
        self.assertGreaterEqual(
            summary["relational_bound_accuracy"], 0.95
        )
        self.assertGreaterEqual(
            summary["reversed_order_dual_success_rate"], 0.95
        )
        self.assertGreaterEqual(summary["minimum_relation_hops"], 2)

    def test_surface_match_attackers_fail(self):
        summary = self.receipt["summary"]
        self.assertLessEqual(summary["exact_key_accuracy"], 0.10)
        self.assertLessEqual(summary["one_hop_alias_accuracy"], 0.10)

    def test_shuffled_relation_graph_fails(self):
        self.assertLessEqual(
            self.receipt["summary"]["shuffled_graph_accuracy"], 0.10
        )

    def test_relation_graph_is_independent_of_reader_and_trust(self):
        state = self.receipt["state_and_binding"]
        self.assertFalse(state["relation_graph_uses_trust"])
        self.assertFalse(
            state["relation_graph_uses_attention_KV_geometry"]
        )

    def test_all_caches_are_immutable(self):
        self.assertTrue(
            all(row["cache_unchanged"] for row in self.receipt["results"])
        )


if __name__ == "__main__":
    unittest.main()
