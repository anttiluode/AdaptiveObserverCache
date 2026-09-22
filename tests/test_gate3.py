import unittest

try:
    import torch
    import transformers  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Gate 3 requires torch + transformers")
class Gate3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gate3_experiment import build_receipt

        cls.receipt = build_receipt()

    def test_test_direction_is_never_rebuilt(self):
        frozen = self.receipt["frozen_from_gate2"]
        self.assertFalse(frozen["observer_direction_recomputed_on_test"])

    def test_transfer_gate(self):
        self.assertTrue(self.receipt["pass"])
        summary = self.receipt["summary"]
        self.assertGreaterEqual(
            summary["transfer_dual_success_rate"], 5 / 7
        )
        self.assertGreaterEqual(
            summary["same_layout_dual_success_rate"], 4 / 5
        )
        self.assertGreaterEqual(
            summary["paraphrase_dual_success_rate"], 1 / 2
        )

    def test_all_test_caches_are_immutable(self):
        self.assertTrue(
            all(row["cache_unchanged"] for row in self.receipt["results"])
        )


if __name__ == "__main__":
    unittest.main()
