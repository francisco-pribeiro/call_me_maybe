from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from src.vocab import VocabHelper
from src.models import FunctionDefinition


MAX_TOKENS: int = 200
START = "START"
KEY_OPEN = "KEY_OPEN"
KEY_CONTENT = "KEY_CONTENT"
KEY_CLOSE = "KEY_CLOSE"
COLON = "COLON"
VALUE_NUMBER = "VALUE_NUMBER"
VALUE_STRING = "VALUE_STRING"
VALUE_CLOSE = "VALUE_CLOSE"
COMMA_OR_END = "COMMA_OR_END"
END = "END"


def get_valid_name_tokens(
        current_prefix: str,
        function_names: list[str],
        vocab: VocabHelper,
        ) -> set[int]:
    valid_token_ids = set()
    for token_id, token_str in vocab.id_to_token.items():
        candidate = current_prefix + token_str
        for f in function_names:
            if f.startswith(candidate) or candidate.startswith(f):
                valid_token_ids.add(token_id)
                break
    return valid_token_ids


def select_function(
        model: Small_LLM_Model,
        input_ids: list[int],
        functions: list[FunctionDefinition],
        vocab: VocabHelper,
        ) -> FunctionDefinition:
    f_names = []
    for f in functions:
        f_names.append(f.name)

    current_prefix = ""

    for _ in range(MAX_TOKENS):
        logits = model.get_logits_from_input_ids(input_ids)
        valid_token_ids: set[int] = get_valid_name_tokens(current_prefix, f_names, vocab)
        masked_logits = [
            logits[i] if i in valid_token_ids else float("-inf")
            for i in range(len(logits))
        ]
        next_token_id = masked_logits.index(max(masked_logits))
        next_token_str = vocab.id_to_token[next_token_id]
        current_prefix += next_token_str
        input_ids = input_ids + [next_token_id]
        if current_prefix in f_names:
            return next(f for f in functions if f.name == current_prefix)
    raise ValueError(f"Could not select a function name within {MAX_TOKENS} tokens")


def extract_parameters(
        model: Small_LLM_Model,
        input_ids: list[int],
        fn_def: FunctionDefinition,
        vocab: VocabHelper,
        ) -> dict[str, str | float]:

    result: dict[str, str | float] = {}
    state = START
    current_key = ""
    current_value = ""
    value_token_ids: list[int] = []
    param_keys = list(fn_def.parameters.keys())
    key_index = 0

    # precompute token sets once
    numeric_tokens = vocab.get_tokens_matching_chars(
        {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "-"}
    )
    quote_tokens = vocab.get_tokens_matching_exact('"')
    open_brace_tokens = vocab.get_tokens_matching_exact("{")
    close_brace_tokens = vocab.get_tokens_matching_exact("}")
    colon_tokens = vocab.get_tokens_matching_exact(":")
    comma_tokens = vocab.get_tokens_matching_exact(",")
    string_tokens = set(vocab.id_to_token.keys()) - quote_tokens

    for _ in range(MAX_TOKENS):
        curr_key = param_keys[key_index]
        logits = model.get_logits_from_input_ids(input_ids)

        # TOP: only determine valid tokens, never change state here
        if state == START:
            valid_tokens = open_brace_tokens

        elif state == KEY_OPEN:
            valid_tokens = quote_tokens

        elif state == KEY_CONTENT:
            valid_tokens = set()
            for token_id, token_str in vocab.id_to_token.items():
                candidate = current_key + token_str
                if curr_key.startswith(candidate) or candidate.startswith(curr_key):
                    valid_tokens.add(token_id)

        elif state == KEY_CLOSE:
            valid_tokens = quote_tokens

        elif state == COLON:
            valid_tokens = colon_tokens

        elif state == VALUE_NUMBER:
            valid_tokens = numeric_tokens | comma_tokens | close_brace_tokens

        elif state == VALUE_STRING:
            valid_tokens = string_tokens

        elif state == VALUE_CLOSE:
            valid_tokens = quote_tokens

        elif state == COMMA_OR_END:
            if key_index < len(param_keys) - 1:
                valid_tokens = comma_tokens
            else:
                valid_tokens = close_brace_tokens

        else:
            break

        # apply mask and pick next token
        masked_logits = [
            logits[i] if i in valid_tokens else float("-inf")
            for i in range(len(logits))
        ]
        next_token_id = masked_logits.index(max(masked_logits))
        next_token_str = vocab.id_to_token[next_token_id]
        input_ids = input_ids + [next_token_id]

        # BOTTOM: all state transitions happen here, after token is picked
        if state == START:
            state = KEY_OPEN

        elif state == KEY_OPEN:
            current_key = ""
            state = KEY_CONTENT

        elif state == KEY_CONTENT:
            current_key += next_token_str
            if current_key == curr_key:
                state = KEY_CLOSE

        elif state == KEY_CLOSE:
            state = COLON

        elif state == COLON:
            current_value = ""
            value_token_ids = []
            param_type = fn_def.parameters[curr_key].type
            state = VALUE_NUMBER if param_type == "number" else VALUE_CLOSE

        elif state == VALUE_CLOSE:
            state = VALUE_STRING

        elif state == VALUE_NUMBER:
            if next_token_str == ",":
                result[curr_key] = float(current_value)
                key_index += 1
                state = KEY_OPEN
            elif next_token_str == "}":
                result[curr_key] = float(current_value)
                return result
            else:
                current_value += next_token_str

        elif state == VALUE_STRING:
            if '"' in next_token_str:
                quote_pos = next_token_str.index('"')
                result[curr_key] = model.decode(value_token_ids)
                remainder = next_token_str[quote_pos + 1:]
                if '}' in remainder:
                    return result
                elif ',' in remainder:
                    key_index += 1
                    current_key = ""
                    state = KEY_CONTENT if '"' in remainder else KEY_OPEN
                else:
                    state = COMMA_OR_END
            else:
                value_token_ids.append(next_token_id)

        elif state == COMMA_OR_END:
            if next_token_str == ",":
                key_index += 1
                state = KEY_OPEN
            else:
                return result

    raise ValueError(f"Could not extract parameters within {MAX_TOKENS} tokens")
