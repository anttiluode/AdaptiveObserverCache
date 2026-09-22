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

    def test_modes_capture_real_sources_not_only_relative_share(self):
        from pretrained_observer import read

        mode_a = read(self.fixture, self.fixture.mode_a.state)
        mode_b = read(self.fixture, self.fixture.mode_b.state)

        self.assertGreaterEqual(mode_a.mass_a, 0.20)
        self.assertGreaterEqual(mode_b.mass_b, 0.20)
        self.assertGreaterEqual(mode_a.source_share_a, 0.80)
        self.assertLessEqual(mode_b.source_share_a, 0.20)
        self.assertNotEqual(mode_a.prediction, mode_b.prediction)

    def test_cache_is_immutable_across_reads(self):
        from pretrained_observer import cache_digest, read

        before = cache_digest(self.fixture.cache)
        for observer in (
            self.fixture.mode_a.state,
            0.0,
            self.fixture.mode_b.state,
        ):
            read(self.fixture, observer)
        self.assertEqual(cache_digest(self.fixture.cache), before)

    def test_gate2_receipt_beats_fixed_and_reset_controls(self):
        from gate2_experiment import build_receipt

        receipt = build_receipt()
        self.assertTrue(receipt["pass"])
        self.assertTrue(
            receipt["checks"]["absolute_source_engagement_pass"]
        )
        self.assertGreaterEqual(receipt["adaptive"]["test_accuracy"], 0.90)
        self.assertGreaterEqual(receipt["adaptive_advantage"], 0.30)
        self.assertLessEqual(receipt["reset_observer_test_accuracy"], 0.60)
        self.assertTrue(receipt["integrity"]["cache_unchanged"])


if __name__ == "__main__":
    unittest.main()
