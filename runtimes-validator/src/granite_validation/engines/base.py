from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from granite_validation.domain.models import EngineInfo


@dataclass
class EngineConfig:
    """Configuration for an engine adapter."""

    mode: Literal["managed", "external"] = "external"
    base_url: str | None = None
    model_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class AbstractEngine(ABC):
    """Interface that all engine adapters must implement.

    Adapters implement raw capabilities. The runner owns lifecycle policy
    (when to call start/stop/health_check based on mode).
    """

    @abstractmethod
    def engine_id(self) -> str:
        """Unique identifier for this engine (e.g. 'ollama', 'vllm')."""

    @abstractmethod
    def start(self, model: str) -> None:
        """Launch the engine process and block until ready."""

    @abstractmethod
    def stop(self) -> None:
        """Terminate the engine process and release resources."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        """Send a chat completion request. Returns a normalized response dict.

        Returns a dict with the following keys:
            role (str): Always "assistant".
            content (str | None): Text content, or None for tool-only responses.
            tool_calls (list[dict] | None): Tool call objects, if any.
            finish_reason (str): One of "stop", "length", "tool_calls".
            usage (dict | None): Token counts with keys prompt_tokens,
                completion_tokens, total_tokens. May be None if the engine
                does not report usage.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the engine is reachable and ready."""

    @abstractmethod
    def get_info(self) -> EngineInfo:
        """Return engine metadata for the report.

        Must be callable at any point in the engine lifecycle, including
        before start(). The runner calls this before start() to determine
        the execution mode, and during lifecycle-failure reporting.
        """
