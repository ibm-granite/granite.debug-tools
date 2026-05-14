from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any, Iterator, NoReturn
from urllib.parse import urlparse

import requests

from runtimes_validator.domain.models import EngineInfo, EngineTimeoutError
from runtimes_validator.engines.base import EngineConfig
from runtimes_validator.engines.openai_compat import OpenAICompatibleEngine
from runtimes_validator.engines.registry import register_engine

logger = logging.getLogger(__name__)

_DEFAULT_STARTUP_TIMEOUT = 300  # vLLM is slow to load models onto GPU
_DEFAULT_STOP_TIMEOUT = 10
_HEALTH_PROBE_TIMEOUT = 10


@register_engine("vllm")
class VllmEngine(OpenAICompatibleEngine):
    """vLLM inference engine adapter.

    Supports both external (connect to running instance) and managed
    (launch and teardown ``vllm serve``) execution modes.
    """

    _DEFAULT_PORT = 8000

    def __init__(self, config: EngineConfig) -> None:
        super().__init__(config)
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_file: io.BufferedRandom | None = None
        self._vllm_version: str | None = None

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
        return "vllm"

    def supported_modalities(self) -> set[str]:
        return {"text", "vision", "speech"}

    # -- Lifecycle --------------------------------------------------------

    def start(self, model: str) -> None:
        if self._config.mode != "managed":
            raise ValueError(
                f"start() called but mode is '{self._config.mode}', expected 'managed'"
            )
        if self._process is not None:
            raise RuntimeError("start() called but engine is already running")

        binary = self._find_binary()
        cmd = self._build_cmd(binary, model)

        # Record the served model so chat() includes it in the request payload.
        self._config.model_id = model

        self._stderr_file = tempfile.TemporaryFile()
        logger.info("Starting vllm serve: %s", " ".join(cmd))

        # Remove VIRTUAL_ENV so vLLM's child processes (EngineCore) don't
        # inherit the *caller's* virtualenv, which can break imports.
        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None)

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr_file,
            env=env,
        )

        try:
            self._wait_until_ready()
            self._fetch_version()
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        if self._process is None:
            return

        stop_timeout = self._config.extra.get("stop_timeout", _DEFAULT_STOP_TIMEOUT)
        logger.info("Stopping vllm serve (pid=%d)", self._process.pid)

        self._process.terminate()
        try:
            self._process.wait(timeout=stop_timeout)
        except subprocess.TimeoutExpired:
            logger.warning("vllm serve did not exit after SIGTERM, sending SIGKILL")
            self._process.kill()
            self._process.wait(timeout=5)
        finally:
            self._process = None
            if self._stderr_file is not None:
                self._stderr_file.close()
                self._stderr_file = None

    # -- Request enrichment ------------------------------------------------

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        try:
            return super().chat(messages, **kwargs)
        except EngineTimeoutError:
            raise
        except requests.RequestException as exc:
            self._raise_with_stderr(exc)

    def chat_stream(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> Iterator[dict[str, Any]]:
        try:
            yield from super().chat_stream(messages, **kwargs)
        except EngineTimeoutError:
            raise
        except requests.RequestException as exc:
            self._raise_with_stderr(exc)

    # -- Info -------------------------------------------------------------

    def health_check(self) -> bool:
        healthy = super().health_check()
        if healthy and self._vllm_version is None:
            self._fetch_version()
        return healthy

    def get_info(self) -> EngineInfo:
        return EngineInfo(
            engine_id=self.engine_id(),
            version=self._vllm_version or "unknown",
            mode=self._config.mode,
            base_url=self._base_url,
        )

    # -- Private helpers (managed mode) ------------------------------------

    def _find_binary(self) -> str:
        path = self._config.extra.get("vllm_bin") or shutil.which("vllm")
        if not path:
            raise RuntimeError(
                "'vllm' binary not found in PATH. "
                "Install vLLM or set 'vllm_bin' in engine config extra."
            )
        return path

    def _build_cmd(self, binary: str, model: str) -> list[str]:
        parsed = urlparse(self._base_url)
        port = parsed.port or self._DEFAULT_PORT

        cmd = [binary, "serve", model, "--port", str(port)]

        # Only pass --host when explicitly configured; otherwise let vLLM
        # use its own default (0.0.0.0).  Deriving the bind address from
        # the *client* base_url (e.g. "localhost") can break vLLM's
        # internal multi-process communication on some platforms.
        bind_host = self._config.extra.get("bind_host")
        if bind_host is not None:
            cmd.extend(["--host", bind_host])

        tensor_parallel_size = self._config.extra.get("tensor_parallel_size")
        if tensor_parallel_size is not None:
            cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])

        gpu_memory_utilization = self._config.extra.get("gpu_memory_utilization")
        if gpu_memory_utilization is not None:
            cmd.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])

        max_model_len = self._config.extra.get("max_model_len")
        if max_model_len is not None:
            cmd.extend(["--max-model-len", str(max_model_len)])

        dtype = self._config.extra.get("dtype")
        if dtype is not None:
            cmd.extend(["--dtype", dtype])

        quantization = self._config.extra.get("quantization")
        if quantization is not None:
            cmd.extend(["--quantization", quantization])

        cmd.extend(self._config.extra.get("server_args", []))

        return cmd

    def _wait_until_ready(self) -> None:
        if self._process is None:
            raise RuntimeError("vllm serve process was not started")

        process = self._process
        timeout = self._config.extra.get("startup_timeout", _DEFAULT_STARTUP_TIMEOUT)
        deadline = time.monotonic() + timeout
        poll_interval = 0.5

        while time.monotonic() < deadline:
            exit_code = process.poll()
            if exit_code is not None:
                stderr = self._read_stderr()
                raise RuntimeError(
                    f"vllm serve exited during startup with code {exit_code}"
                    + (f": {stderr}" if stderr else "")
                    + ". For large models, consider launching vLLM externally "
                    "and running the validation in external mode instead."
                )
            # Cap the probe timeout to remaining time so startup_timeout is a
            # real upper bound even when the socket is blackholed.
            remaining = max(deadline - time.monotonic(), 0.1)
            try:
                resp = requests.get(
                    f"{self._base_url}{self._HEALTH_ENDPOINT}",
                    headers=self._headers or None,
                    timeout=min(remaining, _HEALTH_PROBE_TIMEOUT),
                )
                if resp.status_code == 200:
                    logger.info("vllm serve is ready")
                    return
            except requests.RequestException:
                pass
            time.sleep(poll_interval)

        raise RuntimeError(
            f"vllm serve did not become ready within {timeout}s. "
            "For large models, consider launching vLLM externally and running "
            "the validation in external mode instead."
        )

    def _fetch_version(self) -> None:
        try:
            resp = requests.get(
                f"{self._base_url}/version",
                headers=self._headers or None,
                timeout=5,
            )
            if resp.status_code == 200:
                self._vllm_version = resp.json().get("version", "unknown")
                logger.info("vLLM version: %s", self._vllm_version)
        except Exception:
            logger.warning("Failed to fetch vLLM version", exc_info=True)

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

    def _raise_with_stderr(self, exc: requests.RequestException) -> NoReturn:
        stderr = self._read_stderr()
        if stderr:
            raise requests.RequestException(
                f"{exc}\n\nvllm stderr (last 4096 bytes):\n{stderr}"
            ) from exc
        raise exc
