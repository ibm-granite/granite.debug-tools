from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any, Iterator
from urllib.parse import urlparse

import requests

from runtimes_validator.domain.models import EngineInfo
from runtimes_validator.engines.base import EngineConfig
from runtimes_validator.engines.openai_compat import OpenAICompatibleEngine
from runtimes_validator.engines.registry import register_engine

logger = logging.getLogger(__name__)

_DEFAULT_STARTUP_TIMEOUT = 30
_DEFAULT_PULL_TIMEOUT = 600
_DEFAULT_STOP_TIMEOUT = 10


@register_engine("ollama")
class OllamaEngine(OpenAICompatibleEngine):
    """Ollama inference engine adapter.

    Supports both external (connect to running instance) and managed
    (launch and teardown ``ollama serve``) execution modes.
    """

    _DEFAULT_PORT = 11434
    _HEALTH_ENDPOINT = "/api/version"

    def __init__(self, config: EngineConfig) -> None:
        super().__init__(config)
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_file: io.BufferedRandom | None = None
        self._ollama_version: str | None = None

    def __del__(self) -> None:
        if self._process is not None:
            self._process.kill()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        if self._stderr_file is not None:
            self._stderr_file.close()

    def engine_id(self) -> str:
        return "ollama"

    def supported_modalities(self) -> set[str]:
        # Ollama's OpenAI-compatible endpoint accepts image_url for
        # vision-capable models (llava, granite-vision, llama3.2-vision).
        # It does not accept audio input.
        return {"text", "vision"}

    # -- Lifecycle --------------------------------------------------------

    def start(self, model: str) -> None:
        if self._config.mode != "managed":
            raise ValueError(
                f"start() called but mode is '{self._config.mode}', expected 'managed'"
            )
        if self._process is not None:
            raise RuntimeError("start() called but engine is already running")

        ollama_bin = self._find_binary()
        env = self._build_env()

        cmd = [ollama_bin, "serve"]
        cmd.extend(self._config.extra.get("server_args", []))

        self._stderr_file = tempfile.TemporaryFile()
        logger.info("Starting ollama serve: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr_file,
        )

        try:
            self._wait_until_ready()
            self._ensure_model(model)
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        if self._process is None:
            return

        stop_timeout = self._config.extra.get("stop_timeout", _DEFAULT_STOP_TIMEOUT)
        logger.info("Stopping ollama serve (pid=%d)", self._process.pid)

        self._process.terminate()
        try:
            self._process.wait(timeout=stop_timeout)
        except subprocess.TimeoutExpired:
            logger.warning("ollama serve did not exit after SIGTERM, sending SIGKILL")
            self._process.kill()
            self._process.wait(timeout=5)
        finally:
            self._process = None
            if self._stderr_file is not None:
                self._stderr_file.close()
                self._stderr_file = None

    # -- Info -------------------------------------------------------------

    def health_check(self) -> bool:
        healthy = super().health_check()
        if healthy and self._ollama_version is None:
            self._fetch_version()
        return healthy

    def get_info(self) -> EngineInfo:
        return EngineInfo(
            engine_id=self.engine_id(),
            version=self._ollama_version or "unknown",
            mode=self._config.mode,
            base_url=self._base_url,
        )

    # -- Ollama native API helpers ------------------------------------------

    def _resolve_model(self, model: str | None) -> str:
        """Return *model* if given, otherwise fall back to ``config.model_id``."""
        resolved = model or self._config.model_id
        if resolved is None:
            raise ValueError("No model specified and config.model_id is None")
        return resolved

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        context: list[Any] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: int | None = None,
        timeout: int = 120,
    ) -> dict[str, Any]:
        """Non-streaming call to ``/api/generate``."""
        payload: dict[str, Any] = {
            "model": self._resolve_model(model),
            "prompt": prompt,
            "stream": False,
        }
        if system is not None:
            payload["system"] = system
        if context is not None:
            payload["context"] = context
        if options is not None:
            payload["options"] = options
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        return self._post_json("/api/generate", payload, timeout=timeout)

    def generate_stream(
        self,
        prompt: str,
        *,
        model: str | None = None,
        options: dict[str, Any] | None = None,
        timeout: int = 180,
    ) -> Iterator[dict[str, Any]]:
        """Streaming call to ``/api/generate``. Yields NDJSON chunks."""
        payload: dict[str, Any] = {
            "model": self._resolve_model(model),
            "prompt": prompt,
            "stream": True,
        }
        if options is not None:
            payload["options"] = options

        yield from self._post_stream_ndjson("/api/generate", payload, timeout=timeout)

    def native_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
        timeout: int = 180,
    ) -> dict[str, Any]:
        """Non-streaming call to ``/api/chat``."""
        payload: dict[str, Any] = {
            "model": self._resolve_model(model),
            "messages": messages,
            "stream": False,
        }
        if tools is not None:
            payload["tools"] = tools
        if options is not None:
            payload["options"] = options

        return self._post_json("/api/chat", payload, timeout=timeout)

    def native_chat_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
        timeout: int = 180,
    ) -> Iterator[dict[str, Any]]:
        """Streaming call to ``/api/chat``. Yields NDJSON chunks."""
        payload: dict[str, Any] = {
            "model": self._resolve_model(model),
            "messages": messages,
            "stream": True,
        }
        if tools is not None:
            payload["tools"] = tools
        if options is not None:
            payload["options"] = options

        yield from self._post_stream_ndjson("/api/chat", payload, timeout=timeout)

    def show(
        self,
        *,
        model: str | None = None,
        timeout: int = 30,
    ) -> dict[str, Any]:
        """Fetch model metadata from ``/api/show``."""
        resp = requests.post(
            f"{self._base_url}/api/show",
            json={"model": self._resolve_model(model)},
            headers=self._headers or None,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()

    # -- Private helpers (managed mode) ------------------------------------

    def _find_binary(self) -> str:
        path = self._config.extra.get("ollama_bin") or shutil.which("ollama")
        if not path:
            raise RuntimeError(
                "'ollama' binary not found in PATH. "
                "Install Ollama or set 'ollama_bin' in engine config extra."
            )
        return path

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        parsed = urlparse(self._base_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or self._DEFAULT_PORT
        # Wrap IPv6 addresses in brackets so Ollama can parse host:port unambiguously
        if ":" in host:
            host = f"[{host}]"
        env["OLLAMA_HOST"] = f"{host}:{port}"
        return env

    def _wait_until_ready(self) -> None:
        timeout = self._config.extra.get("startup_timeout", _DEFAULT_STARTUP_TIMEOUT)
        deadline = time.monotonic() + timeout
        poll_interval = 0.5

        while time.monotonic() < deadline:
            exit_code = self._process.poll()
            if exit_code is not None:
                stderr = self._read_stderr()
                raise RuntimeError(
                    f"ollama serve exited during startup with code {exit_code}"
                    + (f": {stderr}" if stderr else "")
                )
            if self.health_check():
                logger.info("ollama serve is ready")
                return
            time.sleep(poll_interval)

        raise RuntimeError(f"ollama serve did not become ready within {timeout}s")

    def _fetch_version(self) -> None:
        try:
            resp = requests.get(
                f"{self._base_url}{self._HEALTH_ENDPOINT}",
                headers=self._headers or None,
                timeout=5,
            )
            if resp.status_code == 200:
                self._ollama_version = resp.json().get("version", "unknown")
                logger.info("Ollama version: %s", self._ollama_version)
        except Exception:
            logger.warning("Failed to fetch ollama version", exc_info=True)

    def _ensure_model(self, model: str) -> None:
        if self._config.extra.get("skip_pull", False):
            logger.info("Skipping model pull (skip_pull=true)")
            return

        pull_timeout = self._config.extra.get("pull_timeout", _DEFAULT_PULL_TIMEOUT)
        logger.info("Ensuring model '%s' is available (pull timeout: %ds)", model, pull_timeout)

        try:
            resp = requests.post(
                f"{self._base_url}/api/pull",
                json={"name": model, "stream": False},
                timeout=(10, pull_timeout),
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to pull model '{model}': {exc}") from exc

    def _read_stderr(self) -> str:
        if self._stderr_file is None:
            return ""
        try:
            self._stderr_file.seek(0, 2)  # seek to end
            size = self._stderr_file.tell()
            offset = max(0, size - 4096)
            self._stderr_file.seek(offset)
            return self._stderr_file.read().decode(errors="replace").strip()
        except Exception:
            return ""
