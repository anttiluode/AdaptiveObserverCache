import unittest

try:
    import torch
    import transformers  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Gate 2 requires torch + transformers")
class Gate2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pretrained_observer import build_fixture

        cls.fixture = build_fixture()

    def test_real_pretrained_geometry_flips_source_read(self):
        from pretrained_observer import read

        plus = read(self.fixture, +1.0)
        minus = read(self.fixture, -1.0)

        self.assertGreaterEqual(plus.source_share_a, 0.80)
        self.assertLessEqual(minus.source_share_a, 0.20)
        self.assertNotEqual(plus.prediction, minus.prediction)
        self.assertLessEqual(self.fixture.perturbation_ratio, 4.0)

    def test_cache_is_immutable_across_reads(self):
        from pretrained_observer import cache_digest, read

        before = cache_digest(self.fixture.cache)
        for observer in (+1.0, 0.0, -1.0, +1.0):
            read(self.fixture, observer)
        self.assertEqual(cache_digest(self.fixture.cache), before)

    def test_gate2_receipt_beats_fixed_and_reset_controls(self):
        from gate2_experiment import build_receipt

        receipt = build_receipt()
        self.assertTrue(receipt["pass"])
        self.assertGreaterEqual(receipt["adaptive"]["test_accuracy"], 0.90)
        self.assertGreaterEqual(receipt["adaptive_advantage"], 0.30)
        self.assertLessEqual(receipt["reset_observer_test_accuracy"], 0.60)
        self.assertTrue(receipt["integrity"]["cache_unchanged"])


if __name__ == "__main__":
    unittest.main()
