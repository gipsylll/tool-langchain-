import json
import os
import sys
import time

import pytest

from data_analyst.codex_transport import run_cli


def test_accepts_complete_plan_without_waiting_for_cli_shutdown(tmp_path):
    plan = {"title": "Сумма", "calculation": "Сумма revenue", "code": "result = 30", "chart": None}
    script = tmp_path / "fake.py"
    event = {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(plan)}}
    script.write_text(
        "import sys, time\n"
        "sys.stdin.read()\n"
        'print(\'{"type":"turn.started"}\', flush=True)\n'
        f"print({json.dumps(event)!r}, flush=True)\n"
        "time.sleep(10)\n"
    )
    stages = []
    start = time.monotonic()
    result = run_cli(
        [sys.executable, str(script)],
        "test",
        os.environ.copy(),
        tmp_path / "plan.json",
        lambda stage, message: stages.append(stage),
        timeout=2,
    )
    assert json.loads(result) == plan
    assert time.monotonic() - start < 2
    assert "waiting_model" in stages


def test_reports_network_failure_without_exposing_raw_diagnostics(tmp_path):
    script = tmp_path / "fake.py"
    script.write_text(
        "import sys\n"
        "sys.stdin.read()\n"
        "print('error sending request Authorization: Bearer DO-NOT-EXPOSE', file=sys.stderr)\n"
        "raise SystemExit(1)\n"
    )
    stages = []
    with pytest.raises(ValueError, match="соединение") as error:
        run_cli(
            [sys.executable, str(script)],
            "test",
            os.environ.copy(),
            tmp_path / "plan.json",
            lambda stage, message: stages.append(message),
            timeout=2,
        )
    assert "DO-NOT-EXPOSE" not in str(error.value)
    assert "DO-NOT-EXPOSE" not in str(stages)


def test_timeout_reports_last_stage_and_kills_process(tmp_path):
    script = tmp_path / "fake.py"
    script.write_text("import sys, time\nsys.stdin.read()\ntime.sleep(10)\n")
    start = time.monotonic()
    with pytest.raises(ValueError, match="Последний этап: подключение"):
        run_cli(
            [sys.executable, str(script)],
            "test",
            os.environ.copy(),
            tmp_path / "plan.json",
            lambda *args: None,
            timeout=0.2,
        )
    assert time.monotonic() - start < 2


def test_timeout_cleans_children_after_parent_exited(tmp_path):
    script = tmp_path / "fake.py"
    script.write_text(
        "import subprocess, sys\n"
        "sys.stdin.read()\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])\n"
    )
    start = time.monotonic()
    with pytest.raises(ValueError, match="Последний этап"):
        run_cli(
            [sys.executable, str(script)],
            "test",
            os.environ.copy(),
            tmp_path / "plan.json",
            lambda *args: None,
            timeout=0.2,
        )
    assert time.monotonic() - start < 2
