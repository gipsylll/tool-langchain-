"""Контролируемый учебный раннер сгенерированного Python-кода.

Не считать полноценной границей безопасности для произвольных атакующих.
Не разрешаем импорт, I/O, сетевые вызовы и доступ к dunder-атрибутам.
Код выполняется в отдельном процессе с таймаутом, минимальным окружением.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Подмножество API Pandas/NumPy, достаточное для анализа продаж.
ALLOWED_ATTRS = {
    "T",
    "abs",
    "agg",
    "all",
    "any",
    "arange",
    "array",
    "astype",
    "between",
    "clip",
    "columns",
    "corr",
    "count",
    "crosstab",
    "cut",
    "describe",
    "diff",
    "drop",
    "drop_duplicates",
    "dropna",
    "dt",
    "duplicated",
    "empty",
    "fillna",
    "first",
    "floor",
    "groupby",
    "head",
    "idxmax",
    "idxmin",
    "iloc",
    "index",
    "is_monotonic_increasing",
    "isin",
    "isna",
    "isnull",
    "items",
    "last",
    "loc",
    "lower",
    "map",
    "max",
    "mean",
    "median",
    "min",
    "month",
    "nan",
    "nlargest",
    "notna",
    "nsmallest",
    "nunique",
    "pct_change",
    "pivot_table",
    "quantile",
    "rename",
    "replace",
    "reset_index",
    "round",
    "shape",
    "size",
    "sort_index",
    "sort_values",
    "std",
    "str",
    "sum",
    "tail",
    "to_datetime",
    "to_frame",
    "to_numpy",
    "to_period",
    "to_string",
    "to_timestamp",
    "to_list",
    "tolist",
    "assign",
    "concat",
    "DataFrame",
    "Series",
    "unstack",
    "div",
    "mul",
    "sub",
    "add",
    "unique",
    "value_counts",
    "var",
    "where",
    "year",
    "day",
    "days",
    "weekday",
}
ALLOWED_CALLS = {
    "print",
    "len",
    "min",
    "max",
    "sum",
    "abs",
    "round",
    "sorted",
    "range",
    "int",
    "float",
    "str",
    "list",
    "dict",
    "enumerate",
    "zip",
}
FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.While,
    ast.For,
    ast.AsyncFor,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Raise,
    ast.Delete,
    ast.Global,
    ast.Nonlocal,
    ast.Yield,
    ast.YieldFrom,
    ast.Await,
    ast.NamedExpr,
)


def validate_code(code: str) -> None:
    if not code.strip() or len(code) > 3500:
        raise ValueError("Код должен содержать от 1 до 3500 символов")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"Неверный синтаксис Python: {exc.msg}") from exc
    if len(list(ast.walk(tree))) > 550:
        raise ValueError("Слишком сложный код")
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise ValueError(f"Конструкция {node.__class__.__name__} запрещена")
        if isinstance(node, ast.Name) and (
            node.id.startswith("_")
            or node.id
            in {
                "open",
                "exec",
                "eval",
                "compile",
                "globals",
                "locals",
                "getattr",
                "setattr",
                "delattr",
                "vars",
                "dir",
                "type",
                "object",
                "input",
                "breakpoint",
                "__import__",
            }
        ):
            raise ValueError(f"Имя {node.id!r} недоступно")
        if isinstance(node, ast.Attribute) and node.attr not in ALLOWED_ATTRS:
            raise ValueError(f"Атрибут {node.attr!r} не разрешён")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id not in ALLOWED_CALLS:
                raise ValueError(f"Вызов функции {node.func.id!r} не разрешён")
            if not isinstance(node.func, (ast.Attribute, ast.Name)):
                raise ValueError("Разрешены только вызовы известных методов и функций")


def execute_code(
    frame: pd.DataFrame,
    code: str,
    timeout: int = 12,
    *,
    prepared_path: Path | None = None,
) -> dict:
    try:
        validate_code(code)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    with tempfile.TemporaryDirectory(prefix="ai_data_analysis_") as dirname:
        folder = Path(dirname)
        data_path = prepared_path or folder / "dataset.arrow"
        if prepared_path is None:
            frame.to_feather(data_path)
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MPLBACKEND": "Agg",
        }
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(Path(__file__).with_name("runner_worker.py")),
                    str(data_path),
                ],
                input=code,
                text=True,
                capture_output=True,
                timeout=timeout,
                cwd=folder,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"Превышен лимит выполнения: {timeout} с"}
    if completed.returncode != 0:
        return {"ok": False, "error": (completed.stderr or "Ошибка процесса")[-2000:]}
    import json

    try:
        return json.loads(completed.stdout)
    except (ValueError, TypeError):
        return {"ok": False, "error": "Worker вернул неверный формат результата"}
