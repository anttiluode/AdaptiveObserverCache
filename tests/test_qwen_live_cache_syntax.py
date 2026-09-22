import ast
import pathlib
import unittest


class LiveQwenSyntaxTests(unittest.TestCase):
    def test_live_cache_harness_parses(self):
        path = pathlib.Path(__file__).resolve().parents[1] / "qwen_observer_live_cache.py"
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


if __name__ == "__main__":
    unittest.main()
