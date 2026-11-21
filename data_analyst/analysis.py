"""Чистые вычисления; агент управляет ими через отдельные tools."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing_extensions import TypedDict


class AnomalyReport(TypedDict):
    method: str
    anomalies: list[dict[str, object]]
    monthly_values: list[dict[str, object]]
    total_found: int


def _ensure_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise ValueError(f"Нет столбца {column!r}. Доступно: {', '.join(map(str, df.columns))}")


def aggregate_chart_data(
    df: pd.DataFrame,
    x: str,
    y: str,
    aggregation: str = "sum",
    time_grain: str = "none",
    color: str | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    _ensure_column(df, x)
    if y != "_rows":
        _ensure_column(df, y)
    if y == x:
        raise ValueError("Для подсчёта строк используйте y='_rows', а не имя колонки X")
    if color:
        _ensure_column(df, color)
    if aggregation not in {"sum", "mean", "count", "median"}:
        raise ValueError("aggregation: sum, mean, count или median")
    if time_grain not in {"none", "day", "week", "month"}:
        raise ValueError("time_grain: none, day, week, month")
    if not 1 <= top_n <= 100:
        raise ValueError("top_n: от 1 до 100")
    data = df[
        [
            x,
            *([y] if y != "_rows" and y != x else []),
            *([color] if color and color not in {x, y} else []),
        ]
    ].copy()
    if time_grain != "none":
        parsed = pd.to_datetime(data[x], errors="coerce")
        if parsed.notna().sum() == 0:
            raise ValueError("Нельзя преобразовать ось X в даты")
        freq = {"day": "D", "week": "W-MON", "month": "M"}[time_grain]
        data[x] = parsed.dt.to_period(freq).dt.to_timestamp()
    groups = [x] + ([color] if color and color != x else [])
    if y == "_rows":
        result = data.groupby(groups, dropna=False).size().reset_index(name="value")
    else:
        if not pd.api.types.is_numeric_dtype(data[y]) and aggregation != "count":
            raise ValueError(f"Столбец {y!r} не числовой")
        grouped = data.groupby(groups, dropna=False)[y]
        values = grouped.sum(min_count=1) if aggregation == "sum" else grouped.agg(aggregation)
        result = values.reset_index(name="value")
    if time_grain == "none":
        # Для категориальных осей сначала выделяем самые весомые категории.
        major = result.groupby(x, dropna=False)["value"].sum().nlargest(top_n).index
        result = result[result[x].isin(major)]
    else:
        latest = result[x].drop_duplicates().sort_values().tail(top_n)
        result = result[result[x].isin(latest)].sort_values(x)
    return result.reset_index(drop=True)


def build_chart(
    grouped: pd.DataFrame,
    x: str,
    chart_type: str,
    title: str,
    color: str | None = None,
) -> go.Figure:
    if chart_type not in {"line", "bar", "scatter", "area"}:
        raise ValueError("chart_type: line, bar, scatter или area")
    maker = {
        "line": px.line,
        "bar": px.bar,
        "scatter": px.scatter,
        "area": px.area,
    }[chart_type]
    fig = (
        maker(grouped, x=x, y="value", color=color, title=title, markers=True)
        if chart_type == "line"
        else maker(grouped, x=x, y="value", color=color, title=title)
    )
    fig.update_layout(
        template="plotly_dark",
        height=390,
        margin=dict(l=20, r=20, t=65, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
    )
    return fig


def detect_mom_anomalies(
    df: pd.DataFrame,
    date_column: str,
    metric: str,
    threshold_pct: float = 25,
    group_by: str | None = None,
) -> AnomalyReport:
    """Сравнить только соседние наблюдаемые календарные месяцы внутри каждой группы."""
    _ensure_column(df, date_column)
    _ensure_column(df, metric)
    if not 0 < threshold_pct <= 1000:
        raise ValueError("Порог изменения должен быть от 0 до 1000%")
    if not pd.api.types.is_numeric_dtype(df[metric]):
        raise ValueError("Метрика должна быть числовой")
    if group_by:
        _ensure_column(df, group_by)
    reserved = {"month", "previous", "change_pct", "comparison_status"}
    if (
        metric in reserved
        or group_by in reserved
        or metric == date_column
        or group_by
        in {
            date_column,
            metric,
        }
    ):
        raise ValueError("Выберите разные колонки без служебных имён month/previous/change_pct")
    local = df[[date_column, metric, *([group_by] if group_by else [])]].copy()
    local["month"] = pd.to_datetime(local[date_column], errors="coerce").dt.to_period("M")
    local = local.dropna(subset=["month"])
    if local.empty:
        raise ValueError("Нет корректных дат")
    keys = ([group_by] if group_by else []) + ["month"]
    monthly = (
        local.groupby(keys, dropna=False, observed=True)[metric].sum(min_count=1).reset_index()
    )
    if group_by:
        groups = monthly.groupby(group_by, dropna=False, observed=True)
        previous_month = groups["month"].shift(1)
        previous_value = groups[metric].shift(1)
    else:
        previous_month = monthly["month"].shift(1)
        previous_value = monthly[metric].shift(1)
    adjacent = previous_month == monthly["month"] - 1
    monthly["previous"] = previous_value.where(adjacent)
    monthly["comparison_status"] = "comparable"
    monthly.loc[previous_month.isna(), "comparison_status"] = "first_month"
    monthly.loc[previous_month.notna() & ~adjacent, "comparison_status"] = "missing_previous_month"
    monthly.loc[adjacent & (previous_value == 0), "comparison_status"] = "zero_base"
    monthly.loc[
        adjacent & (previous_value.isna() | monthly[metric].isna()), "comparison_status"
    ] = "missing_value"
    monthly["change_pct"] = (
        (monthly[metric] - monthly["previous"]) / monthly["previous"].replace(0, float("nan"))
    ) * 100
    found = monthly.loc[monthly["change_pct"].abs() >= threshold_pct].copy()
    found = found.sort_values("change_pct", key=lambda s: s.abs(), ascending=False)
    monthly["month"] = monthly["month"].astype(str)
    found["month"] = found["month"].astype(str)
    fields = ([group_by] if group_by else []) + [
        "month",
        metric,
        "previous",
        "change_pct",
        "comparison_status",
    ]

    def records(frame: pd.DataFrame) -> list[dict[str, object]]:
        selected = frame[fields].round(2).astype(object)
        return selected.where(pd.notna(selected), None).to_dict(orient="records")

    return {
        "method": (
            f"Месячное изменение по модулю >= {threshold_pct:.1f}% "
            "(не причинно-следственная оценка)"
        ),
        "anomalies": records(found.head(35)),
        "monthly_values": records(monthly.tail(35)),
        "total_found": int(len(found)),
    }
