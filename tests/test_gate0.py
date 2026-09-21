import unittest

from adaptive_observer_cache import ObserverState, gate0_cache
from gate0_experiment import (
    A_STATE,
    B_STATE,
    PRESENT,
    paired_observer_receipt,
    switching_world_receipt,
)


class Gate0Tests(unittest.TestCase):
    def test_same_memory_same_present_different_observer(self):
        cache = gate0_cache()
        a = cache.read(PRESENT, A_STATE)
        b = cache.read(PRESENT, B_STATE)

        self.assertEqual(a.selected_provenance, "A")
        self.assertEqual(a.prediction, 1)
        self.assertEqual(b.selected_provenance, "B")
        self.assertEqual(b.prediction, -1)

    def test_surprise_updates_reader_not_cache(self):
        cache = gate0_cache()
        checksum = cache.checksum()

        first = cache.read(PRESENT, A_STATE)
        changed = cache.update(A_STATE, first, truth=-1)
        second = cache.read(PRESENT, changed)

        self.assertEqual(cache.checksum(), checksum)
        self.assertEqual(first.selected_provenance, "A")
        self.assertEqual(second.selected_provenance, "B")
        self.assertEqual(second.prediction, -1)

    def test_correct_read_does_not_move_observer(self):
        cache = gate0_cache()
        first = cache.read(PRESENT, A_STATE)
        unchanged = cache.update(A_STATE, first, truth=1)
        self.assertEqual(unchanged, A_STATE)

    def test_gate0_receipts_pass(self):
        self.assertTrue(paired_observer_receipt()["pass"])
        switching = switching_world_receipt()
        self.assertTrue(switching["pass"])
        self.assertGreaterEqual(switching["adaptive_accuracy"], 0.75)
        self.assertGreaterEqual(switching["adaptive_advantage"], 0.20)
        self.assertTrue(switching["cache_unchanged"])


if __name__ == "__main__":
    unittest.main()
