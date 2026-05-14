"""Shared helpers for multimodal engine-plumbing tests."""

from __future__ import annotations

from typing import Any

from runtimes_validator.domain.models import CheckResult
from runtimes_validator.engines.base import AbstractEngine


def append_shape_checks(response: dict[str, Any], checks: list[CheckResult]) -> None:
    """Append role / content / finish_reason checks for a chat response."""
    checks.append(
        CheckResult(
            name="response_role",
            passed=response.get("role") == "assistant",
            expected="assistant",
            actual=response.get("role"),
        )
    )
    content = response.get("content") or ""
    checks.append(
        CheckResult(
            name="response_content_nonempty",
            passed=bool(content.strip()),
            expected="non-empty content",
            actual=content[:200],
        )
    )
    checks.append(
        CheckResult(
            name="response_finish_reason",
            passed=response.get("finish_reason") in ("stop", "length"),
            expected="stop or length",
            actual=response.get("finish_reason"),
        )
    )


def _prompt_tokens(response: dict[str, Any]) -> int | None:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    value = usage.get("prompt_tokens")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def append_ingestion_check(
    engine: AbstractEngine,
    mm_response: dict[str, Any],
    prompt_text: str,
    media_label: str,
    checks: list[CheckResult],
) -> None:
    """Check that the multimodal payload added tokens vs. a text-only baseline.

    If the engine does not report ``usage.prompt_tokens``, records a failed
    check rather than silently passing — the framework cannot confirm that the
    media was ingested without a token count.
    """
    mm_pt = _prompt_tokens(mm_response)
    if mm_pt is None:
        checks.append(
            CheckResult(
                name=f"{media_label}_ingested_prompt_tokens",
                passed=False,
                detail=(
                    "engine did not report numeric usage.prompt_tokens for "
                    f"{media_label} request; cannot verify {media_label} ingestion"
                ),
            )
        )
        return

    try:
        text_response = engine.chat([{"role": "user", "content": prompt_text}], max_tokens=32)
    except Exception as e:
        checks.append(CheckResult(name="text_baseline_error", passed=False, detail=str(e)))
        return

    text_pt = _prompt_tokens(text_response)
    if text_pt is None:
        checks.append(
            CheckResult(
                name=f"{media_label}_ingested_prompt_tokens",
                passed=False,
                detail=(
                    "engine did not report numeric usage.prompt_tokens for text baseline; "
                    f"cannot verify {media_label} ingestion"
                ),
            )
        )
        return

    checks.append(
        CheckResult(
            name=f"{media_label}_ingested_prompt_tokens",
            passed=mm_pt > text_pt,
            expected=f"> {text_pt} (text-only baseline)",
            actual=mm_pt,
        )
    )
