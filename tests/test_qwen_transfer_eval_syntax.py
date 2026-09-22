import ast
import pathlib
import unittest


class QwenTransferEvalSyntaxTests(unittest.TestCase):
    def test_transfer_eval_parses(self):
        path = pathlib.Path(__file__).resolve().parents[1] / "qwen_observer_transfer_eval.py"
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


if __name__ == "__main__":
    unittest.main()
