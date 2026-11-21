import json
import subprocess
import sys

from data_analyst.cli import build_parser, run_command


def test_execute_reads_code_file_and_forwards_it(tmp_path):
    class WorkspaceStub:
        def call(self, run_id, tool, arguments):
            return run_id, tool, arguments

    code_file = tmp_path / "analysis.py"
    code_file.write_text("result = 42", encoding="utf-8")
    args = build_parser().parse_args(["execute", "run-1", "--code-file", str(code_file)])

    assert run_command(args, WorkspaceStub()) == (
        "run-1",
        "execute_python",
        {"code": "result = 42"},
    )


def test_cli_reports_missing_file_as_json(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "data_analyst.cli",
            "--data-dir",
            str(tmp_path / "data"),
            "load",
            str(tmp_path / "missing.csv"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert json.loads(completed.stderr)["ok"] is False
