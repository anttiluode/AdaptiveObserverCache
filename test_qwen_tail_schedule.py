import ast
import pathlib
import unittest

from qwen_tail_schedule import (
    ARMS,
    classify,
    generation_choice,
    generation_verdict,
    split_logprobs,
    trust_for_index,
)


def block(a_total, b_total, b_tail, b_decision=-0.5):
    return {
        "A": {"total": a_total, "tail": 0.0, "decision": -1.0},
        "B": {"total": b_total, "tail": b_tail, "decision": b_decision},
    }


class TrustScheduleTests(unittest.TestCase):
    def test_arms(self):
        d = 3
        self.assertEqual([trust_for_index("tonic", i, d, -1) for i in range(6)], [-1.0] * 6)
        self.assertEqual([trust_for_index("neutral", i, d, -1) for i in range(6)], [0.0] * 6)
        self.assertEqual(
            [trust_for_index("phasic", i, d, -1) for i in range(6)],
            [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0],
        )

    def test_tonic_and_phasic_share_everything_up_to_decision(self):
        d = 4
        for i in range(d + 1):
            self.assertEqual(
                trust_for_index("tonic", i, d, -0.7),
                trust_for_index("phasic", i, d, -0.7),
            )

    def test_unknown_arm(self):
        with self.assertRaises(ValueError):
            trust_for_index("pulse", 0, 0, -1)

    def test_split(self):
        s = split_logprobs([-0.1, -0.2, -1.0, -0.5, -0.25], 2)
        self.assertAlmostEqual(s["prefix"], -0.3)
        self.assertAlmostEqual(s["decision"], -1.0)
        self.assertAlmostEqual(s["tail"], -0.75)
        self.assertAlmostEqual(s["total"], -2.05)
        with self.assertRaises(ValueError):
            split_logprobs([-1.0], 1)


class ClassifyTests(unittest.TestCase):
    def near_ok(self):
        return {"tonic": block(-1.2, -0.9, -0.3), "phasic": block(-1.2, -0.9, -0.3),
                "neutral": block(-0.01, -11.0, -0.3)}

    def test_no_baseline(self):
        near = self.near_ok()
        near["tonic"] = block(-0.5, -0.9, -0.3)
        far = self.near_ok()
        self.assertEqual(classify(near, far)["verdict"], "NO_BASELINE_CONTROL")

    def test_phasic_rescues(self):
        far = {"tonic": block(-2.6, -3.1, -2.0), "phasic": block(-2.6, -2.0, -0.9),
               "neutral": block(-0.1, -11.0, -0.4)}
        self.assertEqual(classify(self.near_ok(), far)["verdict"], "PHASIC_RESCUES")

    def test_observer_damage_without_flip(self):
        far = {"tonic": block(-2.6, -3.1, -2.0), "phasic": block(-2.6, -2.8, -1.3),
               "neutral": block(-0.1, -11.0, -0.4)}
        self.assertEqual(classify(self.near_ok(), far)["verdict"], "OBSERVER_DAMAGES_TAIL")

    def test_distance_damage(self):
        far = {"tonic": block(-2.6, -3.1, -2.0), "phasic": block(-2.6, -3.0, -1.9),
               "neutral": block(-0.1, -12.5, -1.8)}
        self.assertEqual(classify(self.near_ok(), far)["verdict"], "DISTANCE_DAMAGES_TAIL")

    def test_unresolved(self):
        far = {"tonic": block(-2.6, -3.1, -0.5), "phasic": block(-2.6, -3.0, -0.4),
               "neutral": block(-0.1, -11.0, -0.4)}
        self.assertEqual(classify(self.near_ok(), far)["verdict"], "UNRESOLVED")

    def test_arm_names(self):
        self.assertEqual(ARMS, ("tonic", "phasic", "neutral"))


class GenerationTests(unittest.TestCase):
    def test_choice(self):
        self.assertEqual(generation_choice("The device failed because sensor K drifted.", " valve", " sensor"), "B")
        self.assertEqual(generation_choice("Valve C was obstructed.", " valve", " sensor"), "A")
        self.assertEqual(generation_choice("Sensor K drift or valve C.", " valve", " sensor"), "both")
        self.assertEqual(generation_choice("Unknown.", " valve", " sensor"), "neither")

    def test_verdict(self):
        self.assertEqual(generation_verdict({"tonic": "A"}, {"tonic": "B"}), "NO_BASELINE_GENERATION")
        self.assertEqual(generation_verdict({"tonic": "B"}, {"tonic": "B"}), "GENERATION_CONTROL_SURVIVES")
        self.assertEqual(generation_verdict({"tonic": "B"}, {"tonic": "both"}), "GENERATION_HEDGES")
        self.assertEqual(generation_verdict({"tonic": "B"}, {"tonic": "A"}), "GENERATION_CONTROL_LOST")


class RunnerSyntaxTests(unittest.TestCase):
    def test_runner_parses(self):
        path = pathlib.Path(__file__).resolve().parents[1] / "qwen_observer_phasic_tail.py"
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


if __name__ == "__main__":
    unittest.main()
