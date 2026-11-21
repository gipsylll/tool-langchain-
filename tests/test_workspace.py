import io
import json
import zipfile

import pandas as pd
import pytest

from data_analyst.runner import execute_code
from data_analyst.workspace import Workspace


@pytest.fixture
def case(tmp_path):
    workspace = Workspace(tmp_path)
    session = workspace.add_dataset(
        b"order_id,region,revenue\n1,A,100\n1,A,100\n2,A,100\n3,B,100\n", "sales.csv"
    )
    return workspace, session, workspace.submit(session, "Средний чек по регионам", "codex")


def compute(workspace, run_id):
    return workspace.call(
        run_id,
        "execute_python",
        {
            "code": "result = df.groupby('region').agg(revenue=('revenue','sum'), "
            "orders=('order_id','nunique')).reset_index()\n"
            "result['average'] = result['revenue'] / result['orders']"
        },
    )


def test_chart_uses_exact_computation_and_survives_reopen(case):
    workspace, _, run_id = case
    response = compute(workspace, run_id)
    assert response["ok"] and response["preview"][0]["average"] == 150
    chart = workspace.call(
        run_id, "create_chart", {"result_id": response["result_id"], "x": "region", "y": "average"}
    )
    assert chart["ok"]
    reopened = Workspace(workspace.root)
    artifact = reopened.artifact(run_id, chart["chart_id"])
    assert artifact["payload"]["source_result_id"] == response["result_id"]
    # Plotly may encode numeric arrays in binary. Round trip via figure rather than JSON indexing.
    import plotly.io as pio

    fig = pio.from_json(json.dumps(artifact["payload"]["figure"]))
    assert len(fig.data) == 1
    assert artifact["payload"]["figure"]["data"][0]["x"] == ["A", "B"]


def test_no_cross_run_artifacts_and_no_finish_without_results(case):
    workspace, session, run_id = case
    response = compute(workspace, run_id)
    other = workspace.submit(session, "Другой вопрос", "gemini")
    with pytest.raises(ValueError, match="не принадлежит"):
        workspace.finish(other, "Ответ", [response["result_id"]])
    with pytest.raises(ValueError, match="result_ids"):
        workspace.finish(run_id, "Ответ", [])
    workspace.finish(run_id, "Средний чек A: 150, B: 100.", [response["result_id"]])
    with pytest.raises(ValueError, match="завершён"):
        workspace.call(run_id, "inspect_data", {})


def test_idempotent_request_and_export(case):
    workspace, session, run_id = case
    assert workspace.submit(session, "Средний чек по регионам", "codex", run_id) == run_id
    response = compute(workspace, run_id)
    workspace.finish(run_id, "A: 150", [response["result_id"]])
    with zipfile.ZipFile(io.BytesIO(workspace.export(run_id))) as archive:
        assert "report.md" in archive.namelist()
        assert "trace.json" in archive.namelist()
        assert len([n for n in archive.namelist() if n.endswith(".csv")]) == 1
        assert "150" in archive.read("report.md").decode()


def test_arrow_preserves_ids_nullable_and_timezone():
    frame = pd.DataFrame(
        {
            "id": pd.Series(["001", "002"], dtype="string"),
            "amount": pd.Series([1, None], dtype="Int64"),
            "when": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
        }
    )
    response = execute_code(frame, "result = df['id'].iloc[0]")
    assert response["ok"] and response["result"]["value"] == "001"
    response = execute_code(frame, "result = str(df['when'].dt.year.iloc[0])")
    assert response["result"]["value"] == "2026"


def test_worker_limits_stdout_and_returns_explicit_errors():
    response = execute_code(pd.DataFrame({"a": [1]}), "print('x' * 70000)")
    assert not response["ok"] and "64 KiB" in response["error"]


def test_budget_is_shared_across_instances(case):
    workspace, _, run_id = case
    for _ in range(12):
        Workspace(workspace.root).call(run_id, "inspect_data", {})
    with pytest.raises(ValueError, match="бюджет"):
        workspace.call(run_id, "inspect_data", {})


def test_submit_idempotency_normalizes_question_and_rejects_conflicts(case):
    workspace, session, _ = case
    request_id = workspace.submit(session, "  Выручка?  ", "codex", "request-1")
    assert workspace.submit(session, "  Выручка?  ", "codex", request_id) == request_id
    assert workspace.submit(session, "Выручка?", "codex", request_id) == request_id
    with pytest.raises(ValueError, match="request_id"):
        workspace.submit(session, "Другой вопрос", "codex", request_id)


def test_artifact_lookup_does_not_decode_unrelated_payload(case):
    workspace, _, run_id = case
    result_id = workspace._save_artifact(run_id, "scalar", {"value": 7})
    with workspace.db() as db:
        db.execute(
            "INSERT INTO artifacts VALUES (?, ?, ?, ?)", ("bad", run_id, "scalar", "invalid json")
        )
    assert workspace.artifact(run_id, result_id)["payload"] == {"value": 7}


def test_concurrent_submit_same_request_id(case):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    workspace, session, _ = case
    barrier = Barrier(2)

    def submit():
        other = Workspace(workspace.root)
        barrier.wait()
        return other.submit(session, "Общий вопрос", "codex", "shared-request")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(submit) for _ in range(2)]
        assert [future.result() for future in futures] == ["shared-request"] * 2
