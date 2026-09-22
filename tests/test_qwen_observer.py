import unittest

try:
    import torch
    from qwen_observer import (
        CapturedGeometry,
        SourceSpans,
        apply_rope,
        choose_observer_plan,
        locate_source_spans,
        minimum_norm_query_update,
    )
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Qwen observer geometry tests require PyTorch")
class QwenObserverGeometryTests(unittest.TestCase):
    def test_minimum_update_hits_positive_margin(self):
        q = torch.tensor([0.0, 1.0], dtype=torch.float64)
        d = torch.tensor([1.0, 0.0], dtype=torch.float64)
        corrected, info = minimum_norm_query_update(
            q,
            d,
            trust=1.0,
            target_margin=2.0,
            scaling=1.0,
            max_ratio=10.0,
        )
        self.assertAlmostEqual(info.achieved_gap, 2.0, places=9)
        self.assertGreater(float(corrected[0]), 0.0)
        self.assertFalse(info.capped)

    def test_update_does_not_reduce_already_satisfied_margin(self):
        q = torch.tensor([3.0, 0.0], dtype=torch.float64)
        d = torch.tensor([1.0, 0.0], dtype=torch.float64)
        corrected, info = minimum_norm_query_update(
            q,
            d,
            trust=1.0,
            target_margin=2.0,
            scaling=1.0,
            max_ratio=1.0,
        )
        self.assertTrue(torch.equal(corrected, q))
        self.assertEqual(info.delta_norm, 0.0)

    def test_update_respects_norm_cap(self):
        q = torch.tensor([0.0, 2.0], dtype=torch.float64)
        d = torch.tensor([1.0, 0.0], dtype=torch.float64)
        _corrected, info = minimum_norm_query_update(
            q,
            d,
            trust=1.0,
            target_margin=100.0,
            scaling=1.0,
            max_ratio=0.25,
        )
        self.assertTrue(info.capped)
        self.assertAlmostEqual(info.ratio, 0.25, places=9)

    def test_rope_rotation_matches_quarter_turn(self):
        q = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
        k = q.clone()
        cos = torch.zeros((1, 1, 4))
        sin = torch.ones((1, 1, 4))
        q2, k2 = apply_rope(q, k, cos, sin)
        expected = torch.tensor([[[[0.0, 0.0, 1.0, 0.0]]]])
        self.assertTrue(torch.equal(q2, expected))
        self.assertTrue(torch.equal(k2, expected))

    def test_source_span_locator_uses_offsets(self):
        rendered = "xx ALPHA yy BETA zz"
        offsets = [(i, i + 1) for i in range(len(rendered))]
        spans = locate_source_spans(
            rendered, offsets, "ALPHA", "BETA"
        )
        self.assertEqual(spans.source_a, (3, 8))
        self.assertEqual(spans.source_b, (12, 16))

    def test_plan_prefers_head_that_can_engage_both_sources(self):
        # Two query heads share one KV head. Head 0 can be steered symmetrically;
        # head 1 starts with a large irrelevant component but shares the same keys.
        keys = torch.tensor(
            [
                [2.0, 0.0],
                [1.5, 0.0],
                [-2.0, 0.0],
                [-1.5, 0.0],
                [0.0, 4.0],
            ],
            dtype=torch.float32,
        ).unsqueeze(0)
        capture = CapturedGeometry(
            layer=4,
            query_states=torch.tensor(
                [[0.0, 0.2], [0.0, 4.0]], dtype=torch.float32
            ),
            key_states=keys,
            scaling=1.0,
            num_key_value_groups=2,
        )
        plan = choose_observer_plan(
            {4: capture},
            SourceSpans((0, 2), (2, 4)),
            target_margin=3.0,
            max_ratio=10.0,
            num_heads=1,
        )
        self.assertEqual(plan.heads[0].query_head, 0)
        self.assertGreater(plan.heads[0].symmetric_source_mass, 0.4)


if __name__ == "__main__":
    unittest.main()
