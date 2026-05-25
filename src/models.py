from pydantic import BaseModel


class ParameterSpec(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    returns: ParameterSpec


class FunctionCall(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, str | float]
