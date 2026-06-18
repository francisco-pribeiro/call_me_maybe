from llm_sdk import Small_LLM_Model  # type: ignore
from src.vocab import VocabHelper
from pydantic import ValidationError
from src.models import FunctionDefinition, FunctionCall, PromptItem
from src.constraints import select_function, extract_parameters
from src.prompt import build_name_prompt, build_params_prompt
import argparse
import json
import os
import sys


def main() -> None:

    # 1. create a parser
    parser = argparse.ArgumentParser()

    # 2. define what arguments you accept
    # By default, the program will read input files from the data/input/
    # directory and write output to the data/output/ directory. You
    # can optionally specify custom paths using the --input and --output
    # arguments. For example:
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calling_results.json")
    parser.add_argument("--functions_definition", default="data/input/functions_definition.json")

    # 3. parse what the user actually typed
    args = parser.parse_args()

    # Load prompts
    try:
        with open(args.input) as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"Error: expected a JSON array in: {args.input}")
            sys.exit(1)
        prompts = [PromptItem.model_validate(item).prompt for item in data]
    except FileNotFoundError:
        print(f"Error: file not found: {args.input}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: invalid JSON in: {args.input}")
        sys.exit(1)
    except ValidationError as e:
        print(f"Error: invalid prompt structure in {args.input}: {e}")
        sys.exit(1)

    # Load functions
    try:
        with open(args.functions_definition) as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"Error: expected a JSON array in: {args.functions_definition}")
            sys.exit(1)
        functions = [FunctionDefinition.model_validate(item) for item in data]
    except FileNotFoundError:
        print(f"Error: file not found: {args.functions_definition}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: invalid JSON in: {args.functions_definition}")
        sys.exit(1)
    except ValidationError as e:
        print(f"Error: invalid function definition structure: {e}")
        sys.exit(1)

    model = Small_LLM_Model()
    vocab = VocabHelper(model)
    function_calls = []

    for prompt in prompts:
        try:
            context = build_name_prompt(prompt, functions)
            # return a 2D tensor
            input_ids = model.encode(context).tolist()[0]
            function = select_function(model, input_ids, functions, vocab)
            context_param = build_params_prompt(prompt, functions, function.name)
            input_param_ids = model.encode(context_param).tolist()[0]
            params = extract_parameters(model, input_param_ids, function, vocab)
            function_call = FunctionCall(prompt=prompt, name=function.name, parameters=params)
            function_calls.append(function_call)
        except ValueError as e:
            print(f"Warning: skipping prompt '{prompt}': {e}")
            continue

    out_dir = os.path.dirname(args.output)
    if out_dir:  # skip when output is a bare filename (current directory)
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump([fc.model_dump() for fc in function_calls], f, indent=2)


if __name__ == "__main__":
    main()
