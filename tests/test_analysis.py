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


def test_mom_marks_missing_month_and_zero_base_separately():
    frame = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-03-01", "2026-04-01", "2026-05-01"],
            "revenue": [100, 200, 0, 50],
        }
    )
    report = detect_mom_anomalies(frame, "date", "revenue")
    months = {row["month"]: row for row in report["monthly_values"]}
    assert months["2026-03"]["comparison_status"] == "missing_previous_month"
    assert months["2026-03"]["previous"] is None
    assert months["2026-03"]["change_pct"] is None
    assert months["2026-05"]["comparison_status"] == "zero_base"
    assert months["2026-05"]["change_pct"] is None
    assert [row["month"] for row in report["anomalies"]] == ["2026-04"]


def test_mom_group_gaps_and_missing_values_are_not_zero():
    frame = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-02-01", "2026-03-01", "2026-01-01", "2026-03-01"],
            "group": ["A", "A", "A", "B", "B"],
            "amount": [5, None, 10, 20, 40],
        }
    )
    report = detect_mom_anomalies(frame, "date", "amount", group_by="group")
    rows = {(row["group"], row["month"]): row for row in report["monthly_values"]}
    assert rows[("A", "2026-02")]["amount"] is None
    assert rows[("A", "2026-03")]["comparison_status"] == "missing_value"
    assert rows[("B", "2026-03")]["comparison_status"] == "missing_previous_month"
    assert report["total_found"] == 0


def test_aggregation_preserves_all_missing_as_missing():
    frame = pd.DataFrame({"category": ["A"], "amount": [float("nan")]})
    result = aggregate_chart_data(frame, "category", "amount")
    assert pd.isna(result.iloc[0]["value"])
