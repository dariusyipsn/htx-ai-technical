"""Per-call LLM logging: model, latency, tokens and cost of every chat model call.

`LLMCallLogger` is a LangChain callback handler. `get_llm` attaches it to the model, so
every call through that model is logged, including calls inside chains and
`with_structured_output`. Metadata passed in a call's config (e.g.
`{"metadata": {"field_group": "TopUps"}}`) is included in its log line.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


class LLMCallLogger(BaseCallbackHandler):
    """Logs one line per chat model call. Cost uses the configured $ per million tokens."""

    def __init__(self, input_cost_per_million: float, output_cost_per_million: float) -> None:
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self._starts: dict[UUID, tuple[float, dict[str, Any]]] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._starts[run_id] = (time.perf_counter(), metadata or {})

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        latency, fields = self._finish(run_id)
        message = response.generations[0][0].message  # type: ignore[attr-defined]
        usage = message.usage_metadata or {}
        input_tokens, output_tokens = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        cost = (input_tokens * self.input_cost_per_million + output_tokens * self.output_cost_per_million) / 1e6
        meta = message.response_metadata
        fields |= {
            "latency_s": f"{latency:.2f}",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": usage.get("input_token_details", {}).get("cache_read", 0),
            "reasoning_tokens": usage.get("output_token_details", {}).get("reasoning", 0),
            "cost_usd": f"{cost:.6f}",
            "finish_reason": meta.get("finish_reason") or meta.get("stop_reason"),
        }
        logger.info("llm call %s", _format(fields))

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        latency, fields = self._finish(run_id)
        fields |= {"latency_s": f"{latency:.2f}", "error": f"{type(error).__name__}: {error}"}
        logger.warning("llm call failed %s", _format(fields))

    def _finish(self, run_id: UUID) -> tuple[float, dict[str, Any]]:
        """Latency since the call started, and the fields that identify the call."""
        started, metadata = self._starts.pop(run_id)
        fields = {
            "provider": metadata.get("ls_provider"),
            "model": metadata.get("ls_model_name"),
            **{k: v for k, v in metadata.items() if not k.startswith(("ls_", "lc_"))},  # skip LangChain's own keys
        }
        return time.perf_counter() - started, fields


def _format(fields: dict[str, Any]) -> str:
    return " ".join(f"{k}={v}" for k, v in fields.items())
