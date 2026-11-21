"""Воспроизводимая генерация синтетических продаж, без внешних данных и API."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

PRODUCTS = {
    "Электроника": [
        ("Наушники", 6400, 4400),
        ("Смарт-часы", 12500, 9100),
        ("Клавиатура", 4200, 2700),
        ("Монитор", 19800, 14900),
    ],
    "Дом": [("Лампа", 2400, 1300), ("Плед", 3200, 1900), ("Посуда", 3600, 2000)],
    "Одежда": [("Футболка", 2100, 950), ("Джинсы", 5400, 2900), ("Куртка", 11300, 7300)],
    "Косметика": [("Крем", 1800, 800), ("Шампунь", 950, 450), ("Парфюм", 6800, 3800)],
    "Спорт": [("Коврик", 2500, 1200), ("Гантели", 5100, 3000), ("Бутылка", 1000, 340)],
}
WEIGHTS = [0.38, 0.18, 0.18, 0.14, 0.12]
FIELDS = [
    "order_id",
    "order_date",
    "customer_id",
    "region",
    "channel",
    "category",
    "product",
    "quantity",
    "unit_price",
    "discount_pct",
    "returned",
    "revenue",
    "cost",
    "profit",
]


def generate(path: Path, seed: int = 42, scale: float = 1.0) -> dict:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    start = date(2026, 1, 1)
    finish = date(2026, 10, 31)
    categories = list(PRODUCTS)
    rows = 0
    by_month = Counter()
    order_id = 100000
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        current = start
        while current <= finish:
            weekday_factor = 1.12 if current.weekday() in (4, 5, 6) else 1.0
            # Намеренно моделируем спад продаж электроники в октябре, причина не содержится в CSV.
            july_factor = 1.18 if current.month == 7 else 1.0
            for category, weight in zip(categories, WEIGHTS, strict=True):
                oct_factor = (
                    0.22
                    if (current.month == 10 and category == "Электроника")
                    else (0.92 if current.month == 10 else 1.0)
                )
                expected = 54 * scale * weight * weekday_factor * july_factor * oct_factor
                n_orders = int(expected) + (rng.random() < expected % 1)
                for _ in range(n_orders):
                    order_id += 1
                    customer = f"C{rng.randint(1000, 11500)}"
                    region = rng.choices(
                        ["Москва", "Санкт-Петербург", "Казань", "Екатеринбург", "Новосибирск"],
                        weights=[45, 24, 13, 10, 8],
                    )[0]
                    channel = rng.choices(
                        ["Сайт", "Мобильное приложение", "Маркетплейс"], weights=[45, 35, 20]
                    )[0]
                    # Один заказ может состоять из нескольких позиций.
                    count_items = rng.choices([1, 2, 3], weights=[72, 22, 6])[0]
                    for _ in range(count_items):
                        product, base_price, base_cost = rng.choice(PRODUCTS[category])
                        quantity = rng.choices([1, 2, 3], weights=[87, 11, 2])[0]
                        unit_price = round(base_price * rng.uniform(0.94, 1.07), 2)
                        discount = rng.choices(
                            [0, 0.05, 0.10, 0.15, 0.25], weights=[55, 15, 17, 10, 3]
                        )[0]
                        returned = int(rng.random() < (0.065 if category == "Одежда" else 0.025))
                        revenue = round(quantity * unit_price * (1 - discount) * (1 - returned), 2)
                        cost = round(quantity * base_cost * (1 - returned), 2)
                        writer.writerow(
                            {
                                "order_id": f"O{order_id}",
                                "order_date": current.isoformat(),
                                "customer_id": customer,
                                "region": region,
                                "channel": channel,
                                "category": category,
                                "product": product,
                                "quantity": quantity,
                                "unit_price": unit_price,
                                "discount_pct": discount,
                                "returned": returned,
                                "revenue": revenue,
                                "cost": cost,
                                "profit": round(revenue - cost, 2),
                            }
                        )
                        rows += 1
                        by_month[str(current)[:7]] += revenue
            current += timedelta(days=1)
    return {
        "rows": rows,
        "monthly_revenue": {k: round(v, 2) for k, v in sorted(by_month.items())},
        "path": str(path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/demo_sales.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    print(generate(args.output, seed=args.seed, scale=args.scale))
