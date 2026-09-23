import ast
import pathlib
import unittest


class QwenFirstAskDistanceSyntaxTests(unittest.TestCase):
    def test_first_ask_distance_harness_parses(self):
        path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "qwen_observer_first_ask_distance.py"
        )
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))


if __name__ == "__main__":
    unittest.main()
