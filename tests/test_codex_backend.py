import json
import sys
import time

import pytest

from data_analyst import codex_backend
from data_analyst.workspace import Workspace


@pytest.fixture
def case(tmp_path):
    workspace = Workspace(tmp_path / "data")
    session = workspace.add_dataset(b"region,revenue\nA,10\nA,20\nB,5\n", "sales.csv")
    run_id = workspace.submit(session, "Сделай график выручки по регионам", "codex")
    return workspace, run_id


def wait_for_completion(workspace, run_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if workspace.run(run_id)["status"] in {"completed", "failed"}:
            return workspace.run(run_id)
        time.sleep(0.01)
    raise AssertionError("Фоновая обработка не завершилась")


def test_automatic_codex_runs_immediately_and_publishes_chart(case, tmp_path, monkeypatch):
    workspace, run_id = case
    plan = {
        "title": "Выручка по регионам",
        "calculation": "Сумма revenue для каждого региона.",
        "code": "result = df.groupby('region')['revenue'].sum().reset_index()",
        "chart": {"x": "region", "y": "revenue", "chart_type": "bar", "title": "Выручка"},
    }
    fake = tmp_path / "codex-fake"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys, time\nfrom pathlib import Path\n"
        "assert 'GEMINI_API_KEY' not in os.environ\n"
        "assert '--ignore-user-config' in sys.argv\n"
        "assert sys.argv[sys.argv.index('--sandbox')+1] == 'read-only'\n"
        "assert 'model_reasoning_effort=\"low\"' in sys.argv\n"
        "assert sys.argv[sys.argv.index('--model')+1] == 'gpt-6-luna'\n"
        "assert 'Сделай график выручки по регионам' in sys.stdin.read()\n"
        "time.sleep(0.4)\n"
        f"Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text({json.dumps(plan)!r})\n"
    )
    fake.chmod(0o755)
    monkeypatch.setattr(codex_backend.shutil, "which", lambda name: str(fake))
    monkeypatch.setenv("GEMINI_API_KEY", "must-not-leak")
    start = time.monotonic()
    assert codex_backend.start_codex(workspace, run_id)
    assert time.monotonic() - start < 0.3
    assert workspace.run(run_id)["status"] == "running"
    assert not codex_backend.start_codex(workspace, run_id)
    result = wait_for_completion(workspace, run_id)
    assert result["status"] == "completed", result
    artifacts = workspace.artifacts(run_id)
    table = next(a for a in artifacts if a["kind"] == "table")
    assert table["payload"]["table"]["data"][0]["revenue"] == 30
    chart = next(a for a in artifacts if a["kind"] == "chart")
    assert chart["payload"]["source_result_id"] == table["id"]
    assert workspace.progress(run_id)["effort"] == "low"
    assert workspace.progress(run_id)["stage"] == "completed"


def test_missing_codex_is_explicit_and_does_not_wait(case, monkeypatch):
    workspace, run_id = case
    monkeypatch.setattr(codex_backend.shutil, "which", lambda name: None)
    assert not codex_backend.start_codex(workspace, run_id)
    assert workspace.run(run_id)["status"] == "failed"
    assert "CLI не найден" in workspace.run(run_id)["answer"]


def test_codex_timeout_releases_job(case, monkeypatch):
    workspace, run_id = case
    monkeypatch.setattr(codex_backend.shutil, "which", lambda name: "/fake/codex")

    def timeout(*args, **kwargs):
        raise ValueError("Codex не вернул план за 90 секунд.")

    monkeypatch.setattr(codex_backend, "run_cli", timeout)
    assert codex_backend.start_codex(workspace, run_id)
    result = wait_for_completion(workspace, run_id)
    assert result["status"] == "failed"
    assert "90 секунд" in result["answer"]
    assert workspace.progress(run_id)["stage"] == "failed"


def test_short_model_name_resolves_from_available_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    (tmp_path / "models_cache.json").write_text(
        json.dumps(
            {
                "models": [
                    {"slug": "gpt-example-luna"},
                    {"slug": "gpt-older-luna"},
                ]
            }
        )
    )
    assert codex_backend.resolve_model("luna") == "gpt-example-luna"
    assert codex_backend.resolve_model("other-model") == "other-model"
    assert codex_backend.resolve_model("") == "gpt-6-luna"
