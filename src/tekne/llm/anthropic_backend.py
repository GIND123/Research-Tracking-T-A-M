"""Anthropic Messages API backend.

Design notes that matter for reproducibility:

* The document text is the *last* thing in the user message and the instructions
  are in the system prompt, so the cacheable prefix is stable across documents
  (see :mod:`tekne.llm.cache` for why we care about determinism as much as cost).
* Structured output is requested via ``output_config.format`` where the model
  supports it; we still validate the result ourselves, because a schema-valid
  response can be perfectly hallucinated and only span grounding catches that.
* Failures are returned as ``LLMResponse(error=...)`` rather than raised.  A
  failed model call must degrade the pipeline to abstention, never crash a
  corpus run halfway through.
"""

from __future__ import annotations

import os
from typing import Any

from .backend import LLMBackend, LLMResponse, Usage
from .cache import CallCache

DEFAULT_MODEL = "claude-opus-5"


class AnthropicBackend(LLMBackend):
    name = "anthropic"
    version = "2"

    def __init__(
        self,
        *,
        cache: CallCache | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        timeout: float = 120.0,
        effort: str = "medium",
    ) -> None:
        super().__init__(cache=cache)
        self.effort = effort
        self._client: Any = None
        self._init_error: str | None = None
        try:
            import anthropic
        except ImportError:
            self._init_error = "anthropic SDK not installed (pip install 'tekne[llm]')"
            return
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            # The SDK also resolves `ant auth login` profiles; let it try, and
            # surface the failure on the first call rather than guessing here.
            pass
        try:
            self._client = anthropic.Anthropic(
                api_key=key, max_retries=max_retries, timeout=timeout
            )
        except Exception as exc:  # pragma: no cover - construction rarely fails
            self._init_error = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self._client is not None

    def _complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int,
        json_schema: dict[str, Any] | None,
    ) -> LLMResponse:
        if self._client is None:
            return LLMResponse(text="", model=model, error=self._init_error or "client unavailable")

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            # Instructions are stable across documents; caching the system block
            # is what makes a corpus run affordable.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
            "output_config": {"effort": self.effort},
        }
        if json_schema is not None:
            kwargs["output_config"]["format"] = {
                "type": "json_schema",
                "schema": json_schema,
            }

        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:
            return self._error_response(model, exc)

        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            return LLMResponse(
                text="",
                model=model,
                stop_reason="refusal",
                error=f"refusal: {getattr(details, 'category', None)}",
            )

        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=text,
            model=getattr(response, "model", model),
            usage=Usage(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            ),
            stop_reason=getattr(response, "stop_reason", None),
        )

    @staticmethod
    def _error_response(model: str, exc: Exception) -> LLMResponse:
        try:
            import anthropic
        except ImportError:  # pragma: no cover
            return LLMResponse(text="", model=model, error=str(exc))

        if isinstance(exc, anthropic.RateLimitError):
            label = "rate_limit"
        elif isinstance(exc, anthropic.AuthenticationError):
            label = "auth"
        elif isinstance(exc, anthropic.BadRequestError):
            label = "bad_request"
        elif isinstance(exc, anthropic.APIConnectionError):
            label = "connection"
        elif isinstance(exc, anthropic.APIStatusError):
            label = f"status_{exc.status_code}"
        else:
            label = type(exc).__name__
        return LLMResponse(text="", model=model, error=f"{label}: {exc}")
