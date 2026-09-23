import math
import unittest

from qwen_first_ask_metrics import (
    first_divergence,
    sequence_control_summary,
    split_first_ask_ids,
)


class FirstAskMetricsTests(unittest.TestCase):
    def test_sequence_summary_reports_winner_pair_probability_and_saturation(self):
        row = sequence_control_summary(sum_a=-0.01, sum_b=-6.0)
        self.assertEqual(row["winner"], "A")
        self.assertGreater(row["pair_probability_A"], 0.99)
        self.assertTrue(row["saturated"])
        self.assertAlmostEqual(row["sum_margin_A_minus_B"], 5.99)

    def test_sequence_summary_keeps_unsaturated_close_decision(self):
        row = sequence_control_summary(sum_a=-1.0, sum_b=-1.2)
        self.assertEqual(row["winner"], "A")
        self.assertFalse(row["saturated"])
        self.assertAlmostEqual(
            row["pair_probability_A"],
            1.0 / (1.0 + math.exp(-0.2)),
        )

    def test_first_divergence_returns_common_prefix_and_equal_depth_decision(self):
        divergence = first_divergence(
            [10, 11, 12, 20, 30],
            [10, 11, 12, 21, 31],
        )
        self.assertEqual(divergence["common_prefix"], [10, 11, 12])
        self.assertEqual(divergence["token_a"], 20)
        self.assertEqual(divergence["token_b"], 21)
        self.assertEqual(divergence["index"], 3)

    def test_first_divergence_rejects_prefix_only_candidates(self):
        with self.assertRaises(ValueError):
            first_divergence([1, 2], [1, 2, 3])

    def test_split_first_ask_closes_system_before_question(self):
        source_ids, question_suffix = split_first_ask_ids(
            [101, 102, 999, 201, 202, 999, 301],
            assistant_end_id=999,
        )
        self.assertEqual(source_ids, [101, 102, 999])
        self.assertEqual(question_suffix, [201, 202, 999, 301])

    def test_split_first_ask_requires_system_boundary(self):
        with self.assertRaises(ValueError):
            split_first_ask_ids([1, 2, 3], assistant_end_id=999)


if __name__ == "__main__":
    unittest.main()
