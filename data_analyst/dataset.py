"""Загрузка и компактный обзор пользовательских CSV/XLSX."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

MAX_ROWS = 200_000
MAX_COLUMNS = 80
MAX_UPLOAD_BYTES = 30 * 1024 * 1024


@dataclass
class Dataset:
    name: str
    frame: pd.DataFrame

    @property
    def rows(self) -> int:
        return len(self.frame)


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Таблица пустая")
    if len(frame) > MAX_ROWS or len(frame.columns) > MAX_COLUMNS:
        raise ValueError(f"Максимум {MAX_ROWS:,} строк и {MAX_COLUMNS} колонок")
    frame = frame.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    if not all(frame.columns) or not frame.columns.is_unique:
        raise ValueError("Названия столбцов должны быть непустыми и уникальными")
    # Автораспознавание только явно названных дат; остальные значения не меняем.
    for col in frame.columns:
        if "date" not in col.lower() and "дата" not in col.lower():
            continue
        if pd.api.types.is_datetime64_any_dtype(frame[col]):
            continue
        converted = pd.to_datetime(frame[col], errors="coerce")
        if converted.notna().sum() >= max(1, int(frame[col].notna().sum() * 0.95)):
            frame[col] = converted
    return frame


def load_dataset(contents: bytes, filename: str) -> Dataset:
    if len(contents) > MAX_UPLOAD_BYTES:
        raise ValueError("Файл превышает 30 МБ")
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        try:
            source = contents.decode("utf-8-sig")
        except UnicodeDecodeError:
            source = contents.decode("cp1251")
        try:
            dialect = csv.Sniffer().sniff(source[:4096], delimiters=",;\t")
            sep = dialect.delimiter
        except csv.Error:
            sep = ","
        columns = next(csv.reader(io.StringIO(source), delimiter=sep), [])
        columns = [col.strip() for col in columns]
        _validate_headers(columns)
        id_types = {
            col: "string" for col in columns if col.lower() == "id" or col.lower().endswith("_id")
        }
        frame = pd.read_csv(
            io.StringIO(source),
            sep=sep,
            low_memory=False,
            nrows=MAX_ROWS + 1,
            dtype=id_types,
            names=columns,
            header=0,
        )
    elif suffix == ".xlsx":
        from openpyxl import load_workbook

        with zipfile.ZipFile(io.BytesIO(contents)) as archive:
            if sum(item.file_size for item in archive.infolist()) > 150 * 1024 * 1024:
                raise ValueError("Распакованный XLSX превышает 150 MiB")
        book = load_workbook(io.BytesIO(contents), read_only=True, data_only=True)
        try:
            sheet = book.worksheets[0]
            headers = next(sheet.iter_rows(max_row=1, values_only=True), ())
            columns = [str(col).strip() if col is not None else "" for col in headers]
            _validate_headers(columns)
        finally:
            book.close()
        frame = pd.read_excel(
            io.BytesIO(contents),
            engine="openpyxl",
            nrows=MAX_ROWS + 1,
            names=columns,
            dtype={
                col: "string"
                for col in columns
                if col.lower() == "id" or col.lower().endswith("_id")
            },
        )
    else:
        raise ValueError("Поддерживаются только CSV и XLSX")
    return Dataset(filename, normalize_frame(frame))


def _validate_headers(columns: list[str]) -> None:
    if not columns or not all(columns) or len(columns) != len(set(columns)):
        raise ValueError("Названия столбцов должны быть непустыми и уникальными")
    if len(columns) > MAX_COLUMNS:
        raise ValueError(f"Максимум {MAX_COLUMNS} колонок")


def json_dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def brief_profile(frame: pd.DataFrame, sample_count: int = 3) -> dict:
    info = {
        "rows": len(frame),
        "columns": [
            {
                "name": str(col),
                "type": str(frame[col].dtype),
                "missing": int(frame[col].isna().sum()),
                "unique": int(frame[col].nunique(dropna=True)),
            }
            for col in frame.columns
        ],
        "duplicate_rows": int(frame.duplicated().sum()),
        "sample": frame.head(sample_count)
        .astype(str)
        .map(lambda value: value[:512])
        .to_dict(orient="records"),
    }
    dates = frame.select_dtypes(include=["datetime", "datetimetz"])
    info["date_ranges"] = {
        col: {"min": str(dates[col].min()), "max": str(dates[col].max())} for col in dates.columns
    }
    numeric = frame.select_dtypes(include="number")
    info["numeric_summary"] = numeric.describe().round(2).to_dict() if not numeric.empty else {}
    return info
