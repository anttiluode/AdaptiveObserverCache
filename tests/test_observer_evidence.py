import math
import unittest

from observer_evidence import EvidenceLedger


class EvidenceLedgerTests(unittest.TestCase):
    def test_external_evidence_updates_trust(self):
        ledger = EvidenceLedger()
        row = ledger.apply("A", 1.0, "sensor", "verified")
        self.assertTrue(row.anchored)
        self.assertAlmostEqual(ledger.score, 1.0)
        self.assertAlmostEqual(ledger.anchored_trust, math.tanh(1.0))

    def test_self_prediction_cannot_anchor_itself(self):
        ledger = EvidenceLedger()
        row = ledger.apply("B", 1.0, "self_prediction", "model preferred B")
        self.assertFalse(row.anchored)
        self.assertEqual(row.score_delta, 0.0)
        self.assertEqual(ledger.score, 0.0)
        self.assertEqual(ledger.anchored_trust, 0.0)

    def test_manual_override_is_separate_from_evidence(self):
        ledger = EvidenceLedger()
        ledger.apply("A", 0.5, "tool")
        anchored = ledger.anchored_trust
        ledger.set_manual(-1.0)
        self.assertEqual(ledger.effective_trust, -1.0)
        self.assertAlmostEqual(ledger.anchored_trust, anchored)
        ledger.set_manual(None)
        self.assertAlmostEqual(ledger.effective_trust, anchored)

    def test_round_trip_preserves_receipts(self):
        ledger = EvidenceLedger()
        ledger.apply("A", 0.25, "user_verification", "checked")
        ledger.apply("B", 0.5, "self", "not evidence")
        restored = EvidenceLedger.from_json(ledger.to_json())
        self.assertAlmostEqual(restored.score, ledger.score)
        self.assertEqual(len(restored.receipts), 2)
        self.assertEqual(restored.receipts[1].score_delta, 0.0)

    def test_invalid_weight_is_rejected(self):
        ledger = EvidenceLedger()
        with self.assertRaises(ValueError):
            ledger.apply("A", 1.1, "sensor")


if __name__ == "__main__":
    unittest.main()
