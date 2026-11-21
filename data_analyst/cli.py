from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_analyst.workspace import Workspace


def main():
    parser = argparse.ArgumentParser(description="Локальный AI Data Analyst")
    parser.add_argument("--data-dir", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("sessions")
    commands.add_parser("pending")
    load = commands.add_parser("load")
    load.add_argument("file", type=Path)
    submit = commands.add_parser("submit")
    submit.add_argument("session_id")
    submit.add_argument("question")
    for name in ("inspect", "show", "execute", "chart", "finish", "export"):
        cmd = commands.add_parser(name)
        cmd.add_argument("run_id")
        if name == "execute":
            source = cmd.add_mutually_exclusive_group(required=True)
            source.add_argument("--code")
            source.add_argument("--code-file", type=Path)
        elif name == "chart":
            cmd.add_argument("result_id")
            cmd.add_argument("x")
            cmd.add_argument("y")
            cmd.add_argument("--type", default="bar")
            cmd.add_argument("--title", default="Результат анализа")
        elif name == "finish":
            cmd.add_argument("--answer-file", type=Path, required=True)
            cmd.add_argument("--result-ids", nargs="+", required=True)
        elif name == "export":
            cmd.add_argument("output", type=Path)
    args = parser.parse_args()
    workspace = Workspace(args.data_dir)
    try:
        match args.command:
            case "sessions":
                result = workspace.sessions()
            case "pending":
                result = workspace.runs(pending=True)
            case "load":
                result = {
                    "session_id": workspace.add_dataset(args.file.read_bytes(), args.file.name)
                }
            case "submit":
                result = {"run_id": workspace.submit(args.session_id, args.question, "codex")}
            case "inspect":
                result = workspace.call(args.run_id, "inspect_data", {})
            case "execute":
                code = args.code_file.read_text() if args.code_file else args.code
                result = workspace.call(args.run_id, "execute_python", {"code": code})
            case "chart":
                result = workspace.call(
                    args.run_id,
                    "create_chart",
                    {
                        "result_id": args.result_id,
                        "x": args.x,
                        "y": args.y,
                        "chart_type": args.type,
                        "title": args.title,
                    },
                )
            case "finish":
                result = workspace.finish(
                    args.run_id, args.answer_file.read_text(), args.result_ids
                )
            case "show":
                result = {
                    "run": workspace.run(args.run_id),
                    "events": workspace.events(args.run_id),
                    "artifacts": [
                        {"id": a["id"], "kind": a["kind"]} for a in workspace.artifacts(args.run_id)
                    ],
                }
            case "export":
                args.output.write_bytes(workspace.export(args.run_id))
                result = {"output": str(args.output.resolve())}
    except (ValueError, OSError) as exc:
        parser.exit(1, json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
