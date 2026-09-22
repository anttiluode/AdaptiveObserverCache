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
        from gate6_alias_binding import build_alias_suite

        cls.base, cls.tokenizer, cls.projections = build_alias_suite()

    def test_alias_translation_resolves_both_generations(self):
        from gate5_generic_binding import capture_persistent_key
        from gate6_alias_binding import (
            bind_alias_key,
            build_alias_registry,
        )

        for projection in self.projections:
            family = projection.case.family
            registry = build_alias_registry(
                self.tokenizer,
                family,
                projection.case.generation,
            )
            for target in (family.key0, family.key1):
                key = capture_persistent_key(self.tokenizer, target)
                slot = bind_alias_key(key, projection, registry)
                current = projection.record_keys[slot]
                self.assertNotEqual(key, current)

    def test_generation2_requires_transitive_receipt(self):
        from gate5_generic_binding import capture_persistent_key
        from gate6_alias_binding import (
            bind_alias_key,
            build_alias_registry,
        )

        projection = next(
            p for p in self.projections if p.case.generation == 2
        )
        family = projection.case.family
        key = capture_persistent_key(self.tokenizer, family.key0)
        one_hop = build_alias_registry(
            self.tokenizer,
            family,
            2,
            truncate_to_one_hop=True,
        )
        with self.assertRaises(RuntimeError):
            bind_alias_key(key, projection, one_hop)

    def test_gate6_records_the_frozen_absolute_engagement_failure(self):
        from gate6_experiment import build_receipt

        receipt = build_receipt()
        self.assertFalse(receipt["pass"])
        self.assertTrue(receipt["checks"]["integrity_pass"])

        summary = receipt["summary"]
        self.assertEqual(summary["identity_reads"], 48)
        self.assertEqual(summary["exact_current_key_matches"], 0)
        self.assertGreaterEqual(summary["alias_bound_accuracy"], 0.95)
        self.assertEqual(summary["reversed_order_dual_success_rate"], 1.0)
        self.assertLess(
            summary["transitive_generation_dual_success_rate"], 0.95
        )
        self.assertLessEqual(summary["gate5_exact_key_accuracy"], 0.10)
        self.assertLessEqual(summary["wrong_alias_map_accuracy"], 0.10)
        self.assertLessEqual(
            summary["generation2_one_hop_only_accuracy"], 0.10
        )

        failing = [
            (row["name"], target, read)
            for row in receipt["results"]
            for target, read in row["reads"].items()
            if not read["alias_bound_pass"]
        ]
        self.assertEqual(len(failing), 1)
        name, target, read = failing[0]
        self.assertEqual(name, "package-g2-01")
        self.assertEqual(target, "Liam")
        self.assertEqual(read["resolved_slot"], 1)
        self.assertGreater(read["target_share"], 0.999)
        self.assertLess(read["target_mass"], 0.20)


if __name__ == "__main__":
    unittest.main()
