import pandas as pd

from data_analyst.runner import execute_code, validate_code


def test_safe_groupby_executes():
    df = pd.DataFrame({"category": ["A", "B", "A"], "revenue": [100, 50, 20]})
    result = execute_code(df, "print(df.groupby('category')['revenue'].sum().to_string())")
    assert result["ok"], result
    assert "120" in result["stdout"]


def test_blocks_filesystem_and_import():
    for code in [
        "import os",
        "open('/etc/passwd')",
        "print(df.__class__)",
        "pd.read_csv('/etc/passwd')",
        "while True: pass",
        "print(object)",
    ]:
        try:
            validate_code(code)
        except ValueError:
            continue
        raise AssertionError(f"Unsafe expression accepted: {code}")


def test_worker_limits_multibyte_output_in_bytes():
    response = execute_code(pd.DataFrame({"a": [1]}), "print('я' * 40000)")
    assert not response["ok"]
    assert "64 KiB" in response["error"]


def test_worker_serializes_series_and_nonfinite_scalar():
    frame = pd.DataFrame({"amount": [1, 2]})
    series = execute_code(frame, "result = df['amount']")
    assert series["ok"]
    assert series["result"]["kind"] == "table"
    assert [row["amount"] for row in series["result"]["table"]["data"]] == [1, 2]
    scalar = execute_code(frame, "result = np.nan")
    assert scalar["ok"] and scalar["result"] == {"kind": "scalar", "value": None}


def test_worker_malformed_response_is_rejected(monkeypatch):
    import subprocess

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout='{"ok": true}', stderr=""
        ),
    )
    response = execute_code(pd.DataFrame({"x": [1]}), "result = 1")
    assert response == {"ok": False, "error": "Worker вернул неверный формат результата"}
