"""Tool definition and decorator."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints


def _is_pydantic_model(t: type) -> bool:
    """Check if a type is a Pydantic BaseModel subclass."""
    try:
        from pydantic import BaseModel
        return isinstance(t, type) and issubclass(t, BaseModel)
    except ImportError:
        return False


@dataclass
class ToolDefinition:
    """A tool that an agent can call."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    _pydantic_model: type | None = None  # set when tool takes a single Pydantic param

    def execute(self, input_data: dict[str, Any]) -> Any:
        if self._pydantic_model is not None:
            # Validate and coerce dict into Pydantic model
            model_instance = self._pydantic_model.model_validate(input_data)
            return self.fn(model_instance)
        return self.fn(**input_data)

    def to_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# Map Python type annotations to JSON Schema types
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json(t: type) -> str:
    return _TYPE_MAP.get(t, "string")


def _build_parameters_schema(fn: Callable[..., Any]) -> tuple[dict[str, Any], type | None]:
    """Infer a JSON Schema 'parameters' object from a function's type hints.

    Returns (schema, pydantic_model_or_none).
    If the function takes a single Pydantic model parameter, uses its
    model_json_schema() for richer schema generation.
    """
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)

    # Filter out 'self'
    params = {k: v for k, v in sig.parameters.items() if k != "self"}

    # Check for single-Pydantic-model pattern: def fn(input: MyModel) -> ...
    if len(params) == 1:
        param_name = next(iter(params))
        param_type = hints.get(param_name)
        if param_type is not None and _is_pydantic_model(param_type):
            schema = param_type.model_json_schema()
            # Pydantic's schema already has type, properties, required
            # Remove $defs, title, etc. that LLMs don't need
            clean_schema: dict[str, Any] = {
                "type": "object",
                "properties": schema.get("properties", {}),
            }
            if "required" in schema:
                clean_schema["required"] = schema["required"]
            # Preserve field descriptions from Pydantic Field(description=...)
            return clean_schema, param_type

    # Standard path: infer from individual parameters
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in params.items():
        prop: dict[str, Any] = {}
        if name in hints:
            hint = hints[name]
            if _is_pydantic_model(hint):
                # Nested Pydantic model as one of multiple params
                prop = hint.model_json_schema()
            else:
                prop["type"] = _python_type_to_json(hint)
        else:
            prop["type"] = "string"

        properties[name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema, None


def tool(fn: Callable[..., Any]) -> ToolDefinition:
    """Decorator that turns a function into a ToolDefinition.

    Supports plain parameters and Pydantic models::

        @tool
        def search_web(query: str, max_results: int = 10) -> str:
            \"\"\"Search the web for information.\"\"\"
            return do_search(query, max_results)

        @tool
        def create_user(input: CreateUserInput) -> str:
            \"\"\"Create a new user from validated input.\"\"\"
            return save_user(input.name, input.email)
    """
    name = fn.__name__
    description = (fn.__doc__ or "").strip()
    parameters, pydantic_model = _build_parameters_schema(fn)

    return ToolDefinition(
        name=name,
        description=description,
        parameters=parameters,
        fn=fn,
        _pydantic_model=pydantic_model,
    )
