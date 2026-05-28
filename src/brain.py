from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from src.vocab import VocabHelper

MAX_TOKENS: int = 200


def generate(
        model: Small_LLM_Model,
        input_ids: list[int],
        vocab: VocabHelper,
        ) -> list[int]:
    generated = []

    for _ in range(MAX_TOKENS):
        logits = model.get_logits_from_input_ids(input_ids)
        next_token = logits.index(max(logits))
        generated.append(next_token)
        input_ids = input_ids + [next_token]
        if vocab.id_to_token.get(next_token) == "}":
            break

    return generated
