import json
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


class VocabHelper:
    def __init__(self, model: Small_LLM_Model) -> None:
        vocab_path = model.get_path_to_vocab_file()

        with open(vocab_path, "r") as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def get_tokens_matching_chars(self, allowed_chars: set[str]) -> set[int]:
        allowed_ids = set()
        for token, token_id in self.token_to_id.items():
            if all(c in allowed_chars for c in token):
                allowed_ids.add(token_id)
        return allowed_ids

    def get_tokens_matching_exact(self, allowed_char: str) -> set[int]:
        return {
            token_id for token, token_id in self.token_to_id.items()
            if token == allowed_char
        }
