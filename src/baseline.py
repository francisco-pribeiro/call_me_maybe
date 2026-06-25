from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

MAX_TOKENS: int = 200


def generate_unconstrained(
        model: Small_LLM_Model,
        input_ids: list[int],
        max_tokens: int = MAX_TOKENS,
        ) -> str:
    """Greedily decode with no constraint mask (a baseline for contrast).

    Picks the raw argmax over the full vocabulary at every step, with no
    valid-token masking and no state machine. Stops at the model's EOS
    token or after max_tokens. Shows what the constrained decoder prevents:
    the output may be malformed JSON, an invented function name, or runaway
    text.

    Args:
        model: The language model, queried for next-token logits.
        input_ids: The encoded prompt to generate from.
        max_tokens: Hard cap on generated tokens (safety net for no EOS).

    Returns:
        The decoded generated text, exactly as the model produced it.
    """
    generated: list[int] = []
    eos = model._tokenizer.eos_token_id
    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(input_ids)
        next_id = logits.index(max(logits))   # argmax, no mask
        if next_id == eos:                     # model signalled "done"
            break
        generated.append(next_id)
        input_ids = input_ids + [next_id]
    return str(model.decode(generated))
