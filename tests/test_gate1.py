import unittest

try:
    import torch
except ImportError:  # Core CI deliberately has no torch.
    torch = None


@unittest.skipIf(torch is None, "Gate 1 requires PyTorch")
class Gate1Tests(unittest.TestCase):
    def test_same_cache_different_observer_changes_read(self):
        from frozen_attention_observer import cache_digest, gate1_fixture

        model, cache, present = gate1_fixture()
        before = cache_digest(cache)

        a = model.read(present, cache, +1.0)
        b = model.read(present, cache, -1.0)

        self.assertEqual(a.selected_provenance, "A")
        self.assertEqual(a.prediction, 1)
        self.assertEqual(b.selected_provenance, "B")
        self.assertEqual(b.prediction, -1)
        self.assertEqual(cache_digest(cache), before)

    def test_feedback_changes_next_query_only(self):
        from frozen_attention_observer import gate1_fixture

        model, cache, present = gate1_fixture()
        first = model.read(present, cache, +1.0)
        changed = model.update_observer(+1.0, first, truth=-1)
        second = model.read(present, cache, changed)

        self.assertEqual(changed, -1.0)
        self.assertFalse(torch.equal(first.query, second.query))
        self.assertEqual(second.selected_provenance, "B")
        self.assertEqual(second.prediction, -1)

    def test_parameters_are_frozen(self):
        from frozen_attention_observer import gate1_fixture

        model, _, _ = gate1_fixture()
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_gate1_receipt_passes_best_fixed_attacker(self):
        from gate1_experiment import build_receipt

        receipt = build_receipt()
        self.assertTrue(receipt["pass"])
        self.assertGreaterEqual(receipt["adaptive"]["accuracy"], 0.75)
        self.assertGreaterEqual(receipt["adaptive_advantage"], 0.20)
        self.assertTrue(receipt["integrity"]["pass"])


if __name__ == "__main__":
    unittest.main()
