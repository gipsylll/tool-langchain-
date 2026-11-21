from io import BytesIO

import pandas as pd
import pytest

from data_analyst.dataset import brief_profile, load_dataset


def test_csv_load_and_profile():
    ds = load_dataset(
        b"order_id,order_date,revenue\nA,2026-01-01,100\nA,2026-01-01,50\n", "sales.csv"
    )
    assert ds.rows == 2
    assert pd.api.types.is_datetime64_any_dtype(ds.frame["order_date"])
    assert brief_profile(ds.frame)["rows"] == 2


def test_xlsx_load():
    buffer = BytesIO()
    pd.DataFrame({"name": ["A", "B"], "amount": [1, 2]}).to_excel(buffer, index=False)
    assert load_dataset(buffer.getvalue(), "demo.xlsx").rows == 2


def test_duplicate_columns_rejected():
    with pytest.raises(ValueError):
        load_dataset(b"x,x\n1,2\n", "bad.csv")


def test_unsupported_format_rejected():
    with pytest.raises(ValueError):
        load_dataset(b"{}", "demo.json")


def test_csv_keeps_identifier_leading_zeroes():
    dataset = load_dataset(b"order_id,revenue\n001,10\n002,20\n", "sales.csv")
    assert dataset.frame["order_id"].tolist() == ["001", "002"]


def test_empty_and_duplicate_xlsx_headers_rejected():
    from openpyxl import Workbook

    for header in [["x", "x"], ["x", None]]:
        book = Workbook()
        book.active.append(header)
        book.active.append([1, 2])
        output = BytesIO()
        book.save(output)
        with pytest.raises(ValueError, match="непустыми и уникальными"):
            load_dataset(output.getvalue(), "bad.xlsx")


def test_date_conversion_requires_at_least_95_percent_valid():
    dataset = load_dataset(
        b"order_date,revenue\n2026-01-01,10\n2026-01-02,20\ninvalid,30\n", "sales.csv"
    )
    assert dataset.frame["order_date"].tolist() == ["2026-01-01", "2026-01-02", "invalid"]


def test_invalid_xlsx_reports_readable_error():
    with pytest.raises(ValueError, match="XLSX"):
        load_dataset(b"not an excel file", "bad.xlsx")


def test_semicolon_csv_preserves_identifier():
    dataset = load_dataset(b"order_id;revenue\n001;10\n002;20\n", "sales.csv")
    assert dataset.frame["order_id"].tolist() == ["001", "002"]
    assert dataset.frame["revenue"].tolist() == [10, 20]
