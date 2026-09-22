import ast
import pathlib
import unittest


class QwenDistanceSweepSyntaxTests(unittest.TestCase):
    def test_distance_sweep_harness_parses(self):
        path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "qwen_observer_distance_sweep.py"
        )
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


if __name__ == "__main__":
    unittest.main()
