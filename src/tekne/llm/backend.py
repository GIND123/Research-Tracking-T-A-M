"""Model-call abstraction.

Every LLM-touching stage talks to this interface, never to a vendor SDK, for
three reasons that all bear on the research claims:

* the guard stack has to be testable against a *deliberately hostile* model, so
  we need a backend that emits fabricated spans and prompt-injection payloads on
  demand (:class:`ScriptedBackend`);
* the ablation "pipeline without any LLM stage" has to be a configuration
  change, not a code path (:class:`NullBackend`);
* cost accounting has to be uniform, because cost-per-document is a reported
  number and not an afterthought.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .cache import CallCache, request_key

#: Published per-million-token prices, used for the cost column in results.
#: Update alongside the model list; a wrong number here is a reporting bug.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )

    def cost_usd(self, model: str) -> float:
        rate_in, rate_out = PRICING.get(model, (0.0, 0.0))
        billed_in = self.input_tokens + 0.1 * self.cache_read_tokens + 1.25 * self.cache_write_tokens
        return (billed_in * rate_in + self.output_tokens * rate_out) / 1_000_000


@dataclass(slots=True)
class LLMResponse:
    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    cached: bool = False
    stop_reason: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class LLMBackend(ABC):
    name: str = "backend"
    #: Bumped whenever prompt handling changes, so the cache invalidates.
    version: str = "1"

    def __init__(self, *, cache: CallCache | None = None) -> None:
        self.cache = cache
        self.calls = 0
        self.usage = Usage()

    @abstractmethod
    def _complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int,
        json_schema: dict[str, Any] | None,
    ) -> LLMResponse:
        ...

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 4096,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        payload = {
            "backend": self.name,
            "backend_version": self.version,
            "model": model,
            "system": system,
            "user": user,
            "max_tokens": max_tokens,
            "json_schema": json_schema,
        }
        key = request_key(payload)
        if self.cache:
            hit = self.cache.get(key)
            if hit is not None:
                self.calls += 1
                return LLMResponse(
                    text=hit["text"],
                    model=hit.get("model", model),
                    usage=Usage(**hit.get("usage", {})),
                    cached=True,
                    stop_reason=hit.get("stop_reason"),
                )

        response = self._complete(
            system=system, user=user, model=model, max_tokens=max_tokens, json_schema=json_schema
        )
        self.calls += 1
        self.usage = self.usage + response.usage
        if self.cache and response.ok:
            self.cache.put(
                key,
                model,
                payload,
                {
                    "text": response.text,
                    "model": response.model,
                    "usage": response.usage.__dict__,
                    "stop_reason": response.stop_reason,
                },
            )
        return response

    def cost_usd(self, model: str) -> float:
        return self.usage.cost_usd(model)

    @property
    def available(self) -> bool:
        return True


class NullBackend(LLMBackend):
    """Refuses every call. Used for the no-LLM ablation and when no key is set."""

    name = "null"

    def _complete(self, **_: Any) -> LLMResponse:
        return LLMResponse(text="", model="none", error="no LLM backend configured")

    @property
    def available(self) -> bool:
        return False


class ScriptedBackend(LLMBackend):
    """Returns canned responses. The test harness for the guard stack.

    Responses are matched by substring against the user prompt, so a single
    script can serve a proposer call and the verifier calls that follow it.
    An unmatched prompt returns ``default``, which defaults to an empty JSON
    object rather than an error -- an adversarial backend should look healthy.
    """

    name = "scripted"

    def __init__(
        self,
        script: list[tuple[str, str]] | None = None,
        *,
        default: str = "{}",
        handler: Callable[[str, str], str] | None = None,
    ) -> None:
        super().__init__(cache=None)
        self.script = script or []
        self.default = default
        self.handler = handler
        self.seen: list[tuple[str, str]] = []

    def _complete(self, *, system: str, user: str, model: str, **_: Any) -> LLMResponse:
        self.seen.append((system, user))
        if self.handler is not None:
            text = self.handler(system, user)
        else:
            text = self.default
            for needle, response in self.script:
                if needle in user:
                    text = response
                    break
        return LLMResponse(
            text=text,
            model=model,
            usage=Usage(input_tokens=len(user) // 4, output_tokens=len(text) // 4),
        )


# --- JSON extraction -------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Recover a JSON object from a model response.

    Structured output makes this unnecessary on the happy path, but the repair
    loop needs it when a model wraps JSON in prose or a fence, and the scripted
    backend exercises exactly those cases.
    """
    text = text.strip()
    if not text:
        return None
    for candidate in _json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"items": parsed}
    return None


def _json_candidates(text: str) -> list[str]:
    out = [text]
    fence = _FENCE.search(text)
    if fence:
        out.insert(0, fence.group(1).strip())
    # Outermost brace/bracket pair.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if 0 <= start < end:
            out.append(text[start : end + 1])
    return out
