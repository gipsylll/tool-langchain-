"""Читаем события CLI сразу; в UI передаём только фиксированные безопасные статусы."""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

MAX_OUTPUT = 512 * 1024


def diagnostic(text: str) -> str | None:
    lower = text.lower()
    if any(word in lower for word in ("401", "unauthorized", "not logged in", "refresh token")):
        return "Codex не принял авторизацию. Выполните codex login."
    if "429" in lower or "quota" in lower or "rate limit" in lower:
        return "Достигнут лимит запросов Codex. Повторите позже."
    if "model" in lower and any(
        word in lower for word in ("not supported", "not found", "does not exist")
    ):
        return "Выбранная модель недоступна в Codex. Проверьте настройки модели."
    if any(
        word in lower
        for word in (
            "reconnect",
            "connection",
            "stream disconnected",
            "error sending request",
            "dns",
            "tls",
            "failed to connect",
        )
    ):
        return "Прервано соединение с Codex. CLI пытается восстановить подключение."
    return None


def _stop(process: subprocess.Popen) -> None:
    if os.name != "posix" and process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=2)
    except ProcessLookupError:
        pass


def _file_result(path: Path) -> str | None:
    try:
        if path.is_file() and 0 < path.stat().st_size <= 65_536:
            text = path.read_text(encoding="utf-8")
            value = json.loads(text)
            if isinstance(value, dict) and isinstance(value.get("code"), str):
                return text
    except (OSError, ValueError, UnicodeError):
        pass
    return None


def run_cli(
    command: list[str],
    prompt: str,
    env: dict[str, str],
    output: Path,
    on_stage: Callable[[str, str], None],
    timeout: float = 90,
) -> str:
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    lines = queue.Queue(maxsize=128)
    stopped = threading.Event()

    def reader():
        try:
            while not stopped.is_set():
                line = process.stdout.readline(65_537)
                if not line:
                    break
                while not stopped.is_set():
                    try:
                        lines.put(line, timeout=0.1)
                        break
                    except queue.Full:
                        pass
        except (OSError, ValueError):
            pass

    def writer():
        try:
            process.stdin.write(prompt.encode())
            process.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass

    readers = threading.Thread(target=reader, daemon=True)
    writers = threading.Thread(target=writer, daemon=True)
    readers.start()
    writers.start()
    deadline, total, reason = time.monotonic() + timeout, 0, None
    last_stage = "подключение к Codex"
    try:
        while True:
            # -o может быть готов раньше выхода CLI (например, при зависании завершения процесса).
            result = _file_result(output)
            if result is not None:
                return result
            try:
                line = lines.get(timeout=0.1)
            except queue.Empty:
                line = None
            if line is not None:
                total += len(line)
                if total > MAX_OUTPUT:
                    raise ValueError("Codex вернул слишком большой диагностический поток.")
                text = line.decode("utf-8", errors="replace")
                try:
                    event = json.loads(text)
                except ValueError:
                    event = {}
                if not isinstance(event, dict):
                    event = {}
                kind = event.get("type", "")
                if not isinstance(kind, str):
                    kind = ""
                if kind == "thread.started":
                    on_stage("connecting", "Codex запущен, устанавливаем соединение")
                elif kind == "turn.started":
                    last_stage = "ожидание ответа модели"
                    on_stage("waiting_model", "Запрос принят CLI, ожидаем ответ модели")
                elif kind.startswith("item."):
                    item = event.get("item", {})
                    if isinstance(item, dict) and item.get("type") in {
                        "reasoning",
                        "agent_message",
                    }:
                        last_stage = "получение плана от модели"
                        on_stage("receiving", "Получаем план расчёта от Codex")
                        if kind == "item.completed" and item.get("type") == "agent_message":
                            try:
                                value = json.loads(item.get("text", ""))
                                if isinstance(value, dict) and isinstance(value.get("code"), str):
                                    return item["text"]
                            except (ValueError, TypeError):
                                pass
                safe = diagnostic(text) if not event or kind in {"error", "turn.failed"} else None
                if safe:
                    reason = safe
                    on_stage("connection_error", safe)
            if process.poll() is not None and lines.empty() and not readers.is_alive():
                result = _file_result(output)
                if result is not None:
                    return result
                raise ValueError(
                    reason
                    or "Codex завершился без плана. Ответ CLI не соответствует ожидаемому формату."
                )
            if time.monotonic() >= deadline:
                if reason:
                    raise ValueError(
                        reason.replace(
                            "CLI пытается восстановить подключение.",
                            "Подключение не восстановлено за отведённое время.",
                        )
                    )
                raise ValueError(
                    f"Codex не вернул план за {timeout:g} секунд. Последний этап: {last_stage}."
                )
    finally:
        stopped.set()
        _stop(process)
        readers.join(timeout=0.5)
        writers.join(timeout=0.5)
        for pipe in (process.stdout, process.stdin):
            if pipe is not None:
                pipe.close()
