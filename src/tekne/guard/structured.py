"""Schema-constrained model calls with a bounded repair loop.

The contract between an agent and the blackboard is a Pydantic model.  A model
response that does not validate is not passed on in a degraded form and is not
patched up heuristically: it gets one repair attempt that quotes the validation
error back, and failing that the call abstains.  Abstention is a first-class
outcome throughout the pipeline, so "the extractor could not produce a
well-formed answer here" is information the review queue receives rather than an
exception someone has to catch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..llm.backend import LLMBackend, LLMResponse, parse_json_object

T = TypeVar("T", bound=BaseModel)


@dataclass
class StructuredResult:
    value: BaseModel | None
    attempts: int = 0
    error: str | None = None
    responses: list[LLMResponse] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.value is not None


class StructuredCaller:
    """Issue a model call that must return an instance of a Pydantic model."""

    def __init__(self, backend: LLMBackend, *, max_repairs: int = 1) -> None:
        self.backend = backend
        self.max_repairs = max_repairs
        self.schema_failures = 0

    def call(
        self,
        model_cls: type[T],
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 4096,
    ) -> StructuredResult:
        schema = _json_schema(model_cls)
        prompt = user
        responses: list[LLMResponse] = []
        last_error: str | None = None

        for attempt in range(self.max_repairs + 1):
            response = self.backend.complete(
                system=system,
                user=prompt,
                model=model,
                max_tokens=max_tokens,
                json_schema=schema,
            )
            responses.append(response)
            if not response.ok:
                return StructuredResult(None, attempt + 1, response.error, responses)

            payload = parse_json_object(response.text)
            if payload is None:
                last_error = "response was not a JSON object"
            else:
                try:
                    return StructuredResult(
                        model_cls.model_validate(payload), attempt + 1, None, responses
                    )
                except ValidationError as exc:
                    last_error = _compact_validation_error(exc)

            self.schema_failures += 1
            if attempt >= self.max_repairs:
                break
            prompt = _repair_prompt(user, response.text, last_error or "unknown error")

        return StructuredResult(None, self.max_repairs + 1, last_error, responses)


def _json_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    schema = model_cls.model_json_schema()
    _tighten(schema)
    return schema


def _tighten(node: Any) -> None:
    """Make every object closed, which is what ``strict`` structured output wants."""
    if isinstance(node, dict):
        if node.get("type") == "object" and "additionalProperties" not in node:
            node["additionalProperties"] = False
        for value in node.values():
            _tighten(value)
    elif isinstance(node, list):
        for value in node:
            _tighten(value)


def _compact_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors()[:4]:
        loc = ".".join(str(p) for p in err.get("loc", ()))
        parts.append(f"{loc}: {err.get('msg')}")
    return "; ".join(parts)


def _repair_prompt(original: str, bad_response: str, error: str) -> str:
    return (
        f"{original}\n\n"
        f"--- repair ---\n"
        f"Your previous response could not be used. It was:\n{bad_response[:1500]}\n\n"
        f"The validation error was: {error}\n"
        f"Return only a JSON object matching the schema. Do not add commentary, "
        f"and do not invent values to satisfy required fields -- if a value is "
        f"not supported by the text, omit the whole item."
    )
