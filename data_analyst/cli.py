import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from data_analyst.workspace import Workspace


def build_parser() -> argparse.ArgumentParser:
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
    inspect = commands.add_parser("inspect")
    inspect.add_argument("run_id")
    show = commands.add_parser("show")
    show.add_argument("run_id")

    execute = commands.add_parser("execute")
    execute.add_argument("run_id")
    source = execute.add_mutually_exclusive_group(required=True)
    source.add_argument("--code")
    source.add_argument("--code-file", type=Path)

    chart = commands.add_parser("chart")
    chart.add_argument("run_id")
    chart.add_argument("result_id")
    chart.add_argument("x")
    chart.add_argument("y")
    chart.add_argument("--type", default="bar")
    chart.add_argument("--title", default="Результат анализа")

    finish = commands.add_parser("finish")
    finish.add_argument("run_id")
    finish.add_argument("--answer-file", type=Path, required=True)
    finish.add_argument("--result-ids", nargs="+", required=True)

    export = commands.add_parser("export")
    export.add_argument("run_id")
    export.add_argument("output", type=Path)
    return parser


def run_command(args: argparse.Namespace, workspace: "Workspace") -> Any:
    match args.command:
        case "sessions":
            result = workspace.sessions()
        case "pending":
            result = workspace.runs(pending=True)
        case "load":
            result = {"session_id": workspace.add_dataset(args.file.read_bytes(), args.file.name)}
        case "submit":
            result = {"run_id": workspace.submit(args.session_id, args.question, "codex")}
        case "inspect":
            result = workspace.call(args.run_id, "inspect_data", {})
        case "execute":
            code = args.code_file.read_text(encoding="utf-8") if args.code_file else args.code
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
                args.run_id, args.answer_file.read_text(encoding="utf-8"), args.result_ids
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
        case _:
            raise ValueError(f"Неизвестная команда: {args.command}")
    return result


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    # Help and argument errors should not load pandas or initialize the workspace.
    from data_analyst.workspace import Workspace

    try:
        result = run_command(args, Workspace(args.data_dir))
    except (ValueError, OSError) as exc:
        parser.exit(1, json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
