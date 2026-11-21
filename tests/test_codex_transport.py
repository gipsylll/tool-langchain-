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


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
def test_stops_child_ignoring_sigterm_after_parent_exits(tmp_path):
    ready = tmp_path / "ready"
    marker = tmp_path / "survived"
    child = tmp_path / "child.py"
    child.write_text(
        "import pathlib, signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "pathlib.Path(sys.argv[1]).touch()\n"
        "time.sleep(0.8)\n"
        "pathlib.Path(sys.argv[2]).touch()\n"
    )
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import pathlib, subprocess, sys, time\n"
        "sys.stdin.read()\n"
        "subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2], sys.argv[3]])\n"
        "while not pathlib.Path(sys.argv[2]).exists(): time.sleep(0.01)\n"
        "print('ordinary output', flush=True)\n"
    )
    with pytest.raises(ValueError):
        run_cli(
            [sys.executable, str(parent), str(child), str(ready), str(marker)],
            "test",
            os.environ.copy(),
            tmp_path / "plan.json",
            lambda *args: None,
            timeout=0.3,
        )
    time.sleep(1)
    assert not marker.exists()
