import pandas as pd
import pytest

from data_analyst.analysis import aggregate_chart_data, build_chart, detect_mom_anomalies


@pytest.fixture
def sales():
    return pd.DataFrame(
        {
            "order_date": pd.to_datetime(
                ["2026-08-01", "2026-08-02", "2026-09-01", "2026-09-02", "2026-10-01", "2026-10-02"]
            ),
            "category": ["A", "B", "A", "B", "A", "B"],
            "revenue": [100, 100, 110, 90, 20, 25],
        }
    )


def test_anomaly_detects_october_drop(sales):
    findings = detect_mom_anomalies(sales, "order_date", "revenue", 25)
    assert findings["total_found"] >= 1
    assert any(r["month"] == "2026-10" and r["change_pct"] < -70 for r in findings["anomalies"])


def test_grouped_anomaly(sales):
    result = detect_mom_anomalies(sales, "order_date", "revenue", 25, "category")
    assert result["total_found"] == 2


def test_chart_groups_correctly(sales):
    grouped = aggregate_chart_data(sales, "order_date", "revenue", time_grain="month")
    assert grouped["value"].tolist() == [200, 200, 45]
    assert len(build_chart(grouped, "order_date", "line", "Demo").data) == 1


def test_chart_rejects_unknown_col(sales):
    with pytest.raises(ValueError):
        aggregate_chart_data(sales, "bad", "revenue")
