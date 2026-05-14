from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any
from urllib.parse import urlparse

import requests

from runtimes_validator.domain.models import EngineInfo
from runtimes_validator.engines.base import EngineConfig
from runtimes_validator.engines.openai_compat import OpenAICompatibleEngine
from runtimes_validator.engines.registry import register_engine

logger = logging.getLogger(__name__)

_LIGHTWEIGHT_TIMEOUT = 30
_DEFAULT_STARTUP_TIMEOUT = 120
_DEFAULT_STOP_TIMEOUT = 10


def _parse_audio_data_url(url: str) -> tuple[str, str]:
    """Return ``(base64_data, format)`` parsed from a base64 audio data URL.

    llama-server accepts only ``wav`` or ``mp3`` as the audio format.
    """
    if not url.startswith("data:audio/"):
        raise ValueError(f"llamacpp requires base64 audio data URLs, got: {url[:40]}...")
    header, _, data = url.partition(",")
    if ";base64" not in header:
        raise ValueError("llamacpp requires base64-encoded audio data URLs")
    mime = header[len("data:") :].split(";", 1)[0]
    subtype = mime.split("/", 1)[1].lower() if "/" in mime else ""
    if subtype in ("wav", "x-wav", "wave"):
        fmt = "wav"
    elif subtype in ("mp3", "mpeg", "mpeg3"):
        fmt = "mp3"
    else:
        raise ValueError(f"llamacpp only supports wav/mp3 audio, got: {subtype!r}")
    return data, fmt


@register_engine("llamacpp")
class LlamaCppEngine(OpenAICompatibleEngine):
    """llama.cpp inference engine adapter.

    Supports both external (connect to running instance) and managed
    (launch and teardown ``llama-server``) execution modes.
    """

    _DEFAULT_PORT = 8080

    def __init__(self, config: EngineConfig) -> None:
        super().__init__(config)
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_file: io.BufferedRandom | None = None
        self._server_version: str | None = None

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
        return "llamacpp"

    def supported_modalities(self) -> set[str]:
        return {"text", "vision", "speech"}

    # -- Chat overrides ---------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        return super().chat(
            [self._rewrite_audio_parts(m) for m in messages],
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @staticmethod
    def _rewrite_audio_parts(message: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI-style ``audio_url`` parts into llama-server's
        ``input_audio`` shape. Leaves every other part untouched."""
        content = message.get("content")
        if not isinstance(content, list):
            return message

        rewritten: list[dict[str, Any]] = []
        changed = False
        for part in content:
            if isinstance(part, dict) and part.get("type") == "audio_url":
                url = (part.get("audio_url") or {}).get("url", "")
                data, fmt = _parse_audio_data_url(url)
                rewritten.append(
                    {"type": "input_audio", "input_audio": {"data": data, "format": fmt}}
                )
                changed = True
            else:
                rewritten.append(part)

        if not changed:
            return message
        return {**message, "content": rewritten}

    # -- Lifecycle --------------------------------------------------------

    def start(self, model: str) -> None:
        if self._config.mode != "managed":
            raise ValueError(
                f"start() called but mode is '{self._config.mode}', expected 'managed'"
            )
        if self._process is not None:
            raise RuntimeError("start() called but engine is already running")

        if not os.path.isfile(model):
            raise FileNotFoundError(f"Model file not found: {model}")

        binary = self._find_binary()
        cmd = self._build_cmd(binary, model)

        self._stderr_file = tempfile.TemporaryFile()
        logger.info("Starting llama-server: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr_file,
        )

        try:
            self._wait_until_ready()
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        if self._process is None:
            return

        stop_timeout = self._config.extra.get("stop_timeout", _DEFAULT_STOP_TIMEOUT)
        logger.info("Stopping llama-server (pid=%d)", self._process.pid)

        self._process.terminate()
        try:
            self._process.wait(timeout=stop_timeout)
        except subprocess.TimeoutExpired:
            logger.warning("llama-server did not exit after SIGTERM, sending SIGKILL")
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
        if healthy and self._server_version is None:
            self._fetch_version()
        return healthy

    def get_info(self) -> EngineInfo:
        return EngineInfo(
            engine_id=self.engine_id(),
            version=self._server_version or "unknown",
            mode=self._config.mode,
            base_url=self._base_url,
        )

    def tokenize(
        self,
        content: str,
        *,
        add_special: bool = True,
        parse_special: bool = False,
    ) -> list[int]:
        """Tokenize text via the ``/tokenize`` endpoint."""
        payload: dict[str, Any] = {"content": content}
        if not add_special:
            payload["add_special"] = False
        if parse_special:
            payload["parse_special"] = True

        resp = requests.post(
            f"{self._base_url}/tokenize",
            json=payload,
            headers=self._headers or None,
            timeout=_LIGHTWEIGHT_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["tokens"]

    def detokenize(self, tokens: list[int]) -> str:
        """Detokenize token IDs via the ``/detokenize`` endpoint."""
        resp = requests.post(
            f"{self._base_url}/detokenize",
            json={"tokens": tokens},
            headers=self._headers or None,
            timeout=_LIGHTWEIGHT_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["content"]

    def props(self) -> dict[str, Any]:
        """Fetch server metadata from the ``/props`` endpoint."""
        resp = requests.get(
            f"{self._base_url}/props",
            headers=self._headers or None,
            timeout=_LIGHTWEIGHT_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    # -- Private helpers (managed mode) ------------------------------------

    def _find_binary(self) -> str:
        path = self._config.extra.get("llamacpp_bin") or shutil.which("llama-server")
        if not path:
            raise RuntimeError(
                "'llama-server' binary not found in PATH. "
                "Install llama.cpp or set 'llamacpp_bin' in engine config extra."
            )
        return path

    def _build_cmd(self, binary: str, model: str) -> list[str]:
        parsed = urlparse(self._base_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or self._DEFAULT_PORT

        cmd = [binary, "--model", model, "--host", host, "--port", str(port)]

        n_gpu_layers = self._config.extra.get("n_gpu_layers")
        if n_gpu_layers is not None:
            cmd.extend(["--n-gpu-layers", str(n_gpu_layers)])

        ctx_size = self._config.extra.get("ctx_size")
        if ctx_size is not None:
            cmd.extend(["--ctx-size", str(ctx_size)])

        if self._config.extra.get("jinja"):
            cmd.append("--jinja")

        parallel = self._config.extra.get("parallel")
        if parallel is not None:
            cmd.extend(["--parallel", str(parallel)])

        cmd.extend(self._config.extra.get("server_args", []))

        return cmd

    def _wait_until_ready(self) -> None:
        timeout = self._config.extra.get("startup_timeout", _DEFAULT_STARTUP_TIMEOUT)
        deadline = time.monotonic() + timeout
        poll_interval = 0.5

        while time.monotonic() < deadline:
            exit_code = self._process.poll()
            if exit_code is not None:
                stderr = self._read_stderr()
                raise RuntimeError(
                    f"llama-server exited during startup with code {exit_code}"
                    + (f": {stderr}" if stderr else "")
                )
            if self.health_check():
                logger.info("llama-server is ready")
                return
            time.sleep(poll_interval)

        raise RuntimeError(f"llama-server did not become ready within {timeout}s")

    def _fetch_version(self) -> None:
        try:
            data = self.props()
            build = data.get("build_info")
            if isinstance(build, str):
                self._server_version = build.split("-")[0] if build else "unknown"
            elif isinstance(build, dict):
                self._server_version = build.get("version", "unknown")
            else:
                self._server_version = "unknown"
            logger.info("llama-server version: %s", self._server_version)
        except Exception:
            logger.warning("Failed to fetch llama-server version", exc_info=True)

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
