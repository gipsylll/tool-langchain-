"""Проверочные цифры для преподавателя без LLM/API."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "demo_sales.csv")
months = df["order_date"].str[:7]
sep = df.loc[months == "2026-09", "revenue"].sum()
oct = df.loc[months == "2026-10", "revenue"].sum()
electronic_sep = df.loc[(months == "2026-09") & (df["category"] == "Электроника"), "revenue"].sum()
electronic_oct = df.loc[(months == "2026-10") & (df["category"] == "Электроника"), "revenue"].sum()
print(f"Строк: {len(df):,}; заказов: {df.order_id.nunique():,}")
print(f"Сумма выручки: {df.revenue.sum():,.2f} ₽")
print(f"Средний чек: {df.revenue.sum() / df.order_id.nunique():,.2f} ₽")
print(f"Сентябрь: {sep:,.2f} ₽; октябрь: {oct:,.2f} ₽")
print(f"Снижение: {(oct / sep - 1) * 100:.2f}%")
print(f"Вклад электроники: {(electronic_sep - electronic_oct) / (sep - oct) * 100:.2f}%")
assert len(df) == 22831
assert abs((oct / sep - 1) * 100 + 50.400655253) < 0.005
assert electronic_sep - electronic_oct > 0.9 * (sep - oct)
print("Проверка пройдена")
