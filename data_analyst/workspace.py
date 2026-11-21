"""Общие локальные данные и инструменты для UI, Gemini, CLI и MCP."""

from __future__ import annotations

import io
import json
import os
import sqlite3
import threading
import time
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import plotly.express as px

from data_analyst.dataset import brief_profile, json_dump, load_dataset
from data_analyst.runner import execute_code


def default_data_dir() -> Path:
    return Path(os.getenv("ANALYST_DATA_DIR", Path(__file__).resolve().parents[1] / ".local-data"))


def new_id() -> str:
    return uuid.uuid4().hex


class Workspace:
    def __init__(self, root: Path | None = None):
        self.root = (root or default_data_dir()).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.RLock()
        with self.db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, mode TEXT NOT NULL,
                    question TEXT NOT NULL, status TEXT NOT NULL, answer TEXT NOT NULL DEFAULT '',
                    result_ids TEXT NOT NULL DEFAULT '[]', created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, kind TEXT NOT NULL,
                    payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, name TEXT NOT NULL,
                    arguments TEXT NOT NULL, response TEXT NOT NULL, elapsed REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS run_progress (
                    run_id TEXT PRIMARY KEY, stage TEXT NOT NULL, message TEXT NOT NULL,
                    started REAL NOT NULL, updated REAL NOT NULL,
                    model TEXT NOT NULL, effort TEXT NOT NULL);
            """)

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.root / "workspace.sqlite3", timeout=15)
        try:
            db.row_factory = sqlite3.Row
            with db:
                yield db
        finally:
            db.close()

    def sessions(self) -> list[dict]:
        with self.db() as db:
            return [dict(row) for row in db.execute("SELECT * FROM sessions ORDER BY created DESC")]

    def add_dataset(self, contents: bytes, name: str) -> str:
        dataset = load_dataset(contents, name)
        session_id = new_id()
        dataset.frame.reset_index(drop=True).to_feather(self.root / f"{session_id}.arrow")
        with self.db() as db:
            db.execute("INSERT INTO sessions VALUES (?, ?, ?)", (session_id, name, time.time()))
        return session_id

    def frame(self, session_id: str) -> pd.DataFrame:
        self.session(session_id)
        return pd.read_feather(self.root / f"{session_id}.arrow")

    def session(self, session_id: str) -> dict:
        with self.db() as db:
            row = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            raise ValueError("Сессия не найдена")
        return dict(row)

    def submit(
        self, session_id: str, question: str, mode: str, request_id: str | None = None
    ) -> str:
        self.session(session_id)
        if mode not in {"codex", "gemini"} or not question.strip() or len(question) > 6000:
            raise ValueError("Нужен вопрос до 6000 символов и режим codex/gemini")
        run_id = request_id or new_id()
        with self.db() as db:
            old = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if old:
                if (old["session_id"], old["question"], old["mode"]) != (
                    session_id,
                    question,
                    mode,
                ):
                    raise ValueError("request_id уже использован с другими параметрами")
                return run_id
            db.execute(
                "INSERT INTO runs(id,session_id,mode,question,status,created) VALUES(?,?,?,?,?,?)",
                (run_id, session_id, mode, question.strip(), "pending", time.time()),
            )
        return run_id

    def runs(self, session_id: str | None = None, pending: bool = False) -> list[dict]:
        query, args = "SELECT * FROM runs WHERE 1=1", []
        if session_id:
            query += " AND session_id=?"
            args.append(session_id)
        if pending:
            query += " AND status IN ('pending', 'running') AND mode='codex'"
        with self.db() as db:
            return [dict(r) for r in db.execute(query + " ORDER BY created", args)]

    def run(self, run_id: str) -> dict:
        with self.db() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("Запрос не найден")
        return dict(row)

    def events(self, run_id: str) -> list[dict]:
        with self.db() as db:
            return [
                {
                    **dict(row),
                    "arguments": json.loads(row["arguments"]),
                    "response": json.loads(row["response"]),
                }
                for row in db.execute("SELECT * FROM events WHERE run_id=? ORDER BY id", (run_id,))
            ]

    def claim(self, run_id: str) -> bool:
        with self.db() as db:
            return bool(
                db.execute(
                    "UPDATE runs SET status='running' "
                    "WHERE id=? AND mode='codex' AND status='pending'",
                    (run_id,),
                ).rowcount
            )

    def progress(self, run_id: str) -> dict | None:
        with self.db() as db:
            row = db.execute("SELECT * FROM run_progress WHERE run_id=?", (run_id,)).fetchone()
            return dict(row) if row is not None else None

    def set_progress(
        self,
        run_id: str,
        stage: str,
        message: str,
        model: str = "Codex CLI: модель по умолчанию",
        effort: str = "low",
    ) -> None:
        now = time.time()
        with self.db() as db:
            db.execute(
                "INSERT INTO run_progress VALUES(?,?,?,?,?,?,?) "
                "ON CONFLICT(run_id) DO UPDATE SET stage=excluded.stage, "
                "message=excluded.message,updated=excluded.updated",
                (run_id, stage, message, now, now, model, effort),
            )

    def artifacts(self, run_id: str) -> list[dict]:
        with self.db() as db:
            return [
                {**dict(row), "payload": json.loads(row["payload"])}
                for row in db.execute("SELECT * FROM artifacts WHERE run_id=?", (run_id,))
            ]

    def artifact(self, run_id: str, result_id: str) -> dict:
        for artifact in self.artifacts(run_id):
            if artifact["id"] == result_id:
                return artifact
        raise ValueError("Результат не принадлежит этому запросу")

    def _save_artifact(self, run_id: str, kind: str, payload: dict) -> str:
        result_id = new_id()
        with self.db() as db:
            db.execute(
                "INSERT INTO artifacts VALUES(?,?,?,?)",
                (result_id, run_id, kind, json_dump(payload)),
            )
        return result_id

    def _begin_tool(self, run_id: str, name: str, arguments: dict) -> tuple[dict, int]:
        # Резервируем бюджет до вычисления: работает и между процессами CLI / UI / MCP.
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            run = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if run is None or run["status"] not in {"pending", "running"}:
                raise ValueError("Запрос не найден или уже завершён")
            count = db.execute("SELECT count(*) FROM events WHERE run_id=?", (run_id,)).fetchone()[
                0
            ]
            if count >= 12:
                raise ValueError("Достигнут бюджет: 12 вызовов инструментов")
            db.execute("UPDATE runs SET status='running' WHERE id=?", (run_id,))
            event = db.execute(
                "INSERT INTO events(run_id,name,arguments,response,elapsed) VALUES(?,?,?,?,0)",
                (run_id, name, json_dump(arguments), '{"status":"running"}'),
            )
            return dict(run), event.lastrowid

    def call(self, run_id: str, name: str, arguments: dict) -> dict:
        with self._lock:
            run, event_id = self._begin_tool(run_id, name, arguments)
            start = time.monotonic()
            try:
                if name == "inspect_data":
                    response = {"ok": True, **brief_profile(self.frame(run["session_id"]))}
                elif name == "execute_python":
                    response = execute_code(
                        pd.DataFrame(),
                        arguments["code"],
                        prepared_path=self.root / f"{run['session_id']}.arrow",
                    )
                    value = response.pop("result", None)
                    if response["ok"] and value is not None:
                        result_id = self._save_artifact(run_id, value["kind"], value)
                        response["result_id"] = result_id
                        if value["kind"] == "table":
                            records = value["table"]["data"]
                            response.update(
                                rows=len(records),
                                preview=records[:15],
                                truncated=len(records) > 15,
                                columns=[f["name"] for f in value["table"]["schema"]["fields"]],
                            )
                        else:
                            response["value"] = value["value"]
                    elif response["ok"]:
                        response["warning"] = "Для таблицы/графика присвойте итог переменной result"
                elif name == "create_chart":
                    response = self._chart(run_id, **arguments)
                else:
                    raise ValueError("Неизвестный инструмент")
            except Exception as exc:
                response = {"ok": False, "error": str(exc)[:1200]}
            with self.db() as db:
                db.execute(
                    "UPDATE events SET response=?,elapsed=? WHERE id=?",
                    (json_dump(response), time.monotonic() - start, event_id),
                )
            return response

    def _chart(
        self,
        run_id: str,
        result_id: str,
        x: str,
        y: str,
        chart_type: str = "bar",
        title: str = "Результат анализа",
    ) -> dict:
        artifact = self.artifact(run_id, result_id)
        if artifact["kind"] != "table":
            raise ValueError("Для графика нужен табличный result_id")
        data = pd.read_json(io.StringIO(json_dump(artifact["payload"]["table"])), orient="table")
        if x not in data or y not in data or not pd.api.types.is_numeric_dtype(data[y]):
            raise ValueError("Проверьте оси: X существует, Y должен быть числовым")
        if chart_type not in {"bar", "line", "scatter", "area"}:
            raise ValueError("Тип графика: bar, line, scatter, area")
        if len(data) > 5000:
            raise ValueError("Сначала сократите результат до 5000 точек явно в execute_python")
        fig = getattr(px, chart_type)(data, x=x, y=y, title=title)
        fig.update_layout(
            template="plotly_white", height=380, margin={"l": 20, "r": 20, "t": 55, "b": 25}
        )
        chart_id = self._save_artifact(
            run_id,
            "chart",
            {
                "figure": json.loads(fig.to_json()),
                "source_result_id": result_id,
                "title": title,
            },
        )
        return {
            "ok": True,
            "chart_id": chart_id,
            "source_result_id": result_id,
            "points": len(data),
        }

    def finish(self, run_id: str, answer: str, result_ids: list[str]) -> dict:
        if not answer.strip() or len(answer) > 30_000:
            raise ValueError("Нужен ответ до 30 000 символов")
        if not result_ids:
            raise ValueError("Укажите result_ids вычислений, на которых основан ответ")
        for result_id in result_ids:
            if self.artifact(run_id, result_id)["kind"] not in {"table", "scalar"}:
                raise ValueError("В ответе нужны ID вычислений, а не графиков")
        with self.db() as db:
            changed = db.execute(
                "UPDATE runs SET status='completed',answer=?,result_ids=? "
                "WHERE id=? AND status IN ('pending','running')",
                (answer.strip(), json_dump(result_ids), run_id),
            ).rowcount
        if not changed:
            raise ValueError("Запрос уже завершён")
        return {"ok": True, "run_id": run_id, "status": "completed"}

    def fail(self, run_id: str, error: str) -> None:
        with self.db() as db:
            db.execute(
                "UPDATE runs SET status='failed',answer=? WHERE id=? AND status!='completed'",
                (error[:2000], run_id),
            )

    def export(self, run_id: str) -> bytes:
        run = self.run(run_id)
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("report.md", f"# {run['question']}\n\n{run['answer']}\n")
            archive.writestr("trace.json", json_dump(self.events(run_id)))
            for artifact in self.artifacts(run_id):
                payload = artifact["payload"]
                if artifact["kind"] == "table":
                    frame = pd.read_json(io.StringIO(json_dump(payload["table"])), orient="table")
                    archive.writestr(f"{artifact['id']}.csv", frame.to_csv(index=False))
                elif artifact["kind"] == "chart":
                    import plotly.io as pio

                    archive.writestr(
                        f"{artifact['id']}.html",
                        pio.to_html(payload["figure"], include_plotlyjs=True),
                    )
                else:
                    archive.writestr(f"{artifact['id']}.json", json_dump(payload))
        return output.getvalue()
