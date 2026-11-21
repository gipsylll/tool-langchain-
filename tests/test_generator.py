import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_demo_data.py"


def test_generator_is_deterministic_and_consistent(tmp_path):
    spec = importlib.util.spec_from_file_location("demo_gen", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    result = module.generate(first, seed=42, scale=0.08)
    module.generate(second, seed=42, scale=0.08)
    assert first.read_bytes() == second.read_bytes()
    df = pd.read_csv(first)
    assert result["rows"] == len(df)
    assert df["order_id"].nunique() < len(df)  # заказы с несколькими строками
    assert ((df["revenue"] - df["cost"] - df["profit"]).abs() <= 0.011).all()
    assert (df.loc[df["returned"] == 1, "revenue"] == 0).all()
    assert df["order_date"].str.startswith("2026-").all()


def test_deliberate_october_anomaly_in_real_dataset():
    csv = Path(__file__).resolve().parents[1] / "data" / "demo_sales.csv"
    df = pd.read_csv(csv)
    month = df["order_date"].str[:7]
    september = df.loc[month == "2026-09", "revenue"].sum()
    october = df.loc[month == "2026-10", "revenue"].sum()
    assert october < september * 0.7
    sep_e = df.loc[(month == "2026-09") & (df["category"] == "Электроника"), "revenue"].sum()
    oct_e = df.loc[(month == "2026-10") & (df["category"] == "Электроника"), "revenue"].sum()
    assert (sep_e - oct_e) / (september - october) > 0.8
