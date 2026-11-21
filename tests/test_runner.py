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
