from src.models import FunctionDefinition


def build_name_prompt(prompt: str, functions: list[FunctionDefinition]) -> str:
    """Build an instruction string that primes the model to output a function name.

    Args:
        prompt: The user request.
        functions: The list of candidate function definitions to include as context.

    Returns:
        A formatted instruction string ending with an open quote to prime generation.
    """
    instructions: str = (
        f'Given these functions: {functions} For the request: {prompt} '
        'Reply with only the function name. Function: "'
    )
    return instructions


def build_params_prompt(
        prompt: str,
        functions: list[FunctionDefinition],
        function_name: str,
        ) -> str:
    """Build an instruction string that primes the model to output parameters as JSON.

    Args:
        prompt: The user request.
        functions: The list of candidate function definitions to include as context.
        function_name: The already-selected function name.

    Returns:
        A formatted instruction string ending with an open brace to prime JSON generation.
    """
    instructions: str = (
        f'Given these functions: {functions} For the request: {prompt} '
        f'The function to call is: {function_name} '
        'Reply with only the parameters as JSON. Parameters: {"'
    )
    return instructions
