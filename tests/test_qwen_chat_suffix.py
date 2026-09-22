import unittest

from qwen_chat_suffix import flat_token_ids, user_turn_suffix_ids


class MappingTokenizer:
    unk_token_id = -1
    eos_token_id = 0

    def convert_tokens_to_ids(self, token):
        if token == "<|im_end|>":
            return 99
        return -1

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking=False,
    ):
        self.assert_shape(messages, add_generation_prompt)
        if len(messages) == 3:
            return {"input_ids": [10, 11, 99]}
        return {"input_ids": [10, 11, 99, 20, 21]}

    @staticmethod
    def assert_shape(messages, add_generation_prompt):
        if len(messages) == 3:
            assert not add_generation_prompt
        else:
            assert len(messages) == 4
            assert add_generation_prompt


class ChatSuffixTests(unittest.TestCase):
    def test_mapping_output_is_flattened(self):
        self.assertEqual(
            flat_token_ids({"input_ids": [1, 2, 3]}),
            [1, 2, 3],
        )

    def test_single_batch_row_is_flattened(self):
        self.assertEqual(
            flat_token_ids({"input_ids": [[1, 2, 3]]}),
            [1, 2, 3],
        )

    def test_live_suffix_accepts_batchencoding_style_mapping(self):
        tokenizer = MappingTokenizer()
        self.assertEqual(
            user_turn_suffix_ids(tokenizer, "Why did it fail?"),
            [20, 21],
        )

    def test_mapping_without_input_ids_fails_cleanly(self):
        with self.assertRaises(ValueError):
            flat_token_ids({"attention_mask": [1, 1]})


if __name__ == "__main__":
    unittest.main()
