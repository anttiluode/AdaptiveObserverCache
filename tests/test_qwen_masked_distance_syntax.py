import ast
import pathlib
import unittest


class QwenMaskedDistanceSyntaxTests(unittest.TestCase):
    def test_masked_distance_harness_parses(self):
        path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "qwen_observer_masked_distance.py"
        )
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


if __name__ == "__main__":
    unittest.main()
