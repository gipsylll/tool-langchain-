"""Ограниченный учебный worker; AST не является промышленной песочницей."""

from __future__ import annotations

import contextlib
import io
import json
import sys

# Лимиты ставятся в дочернем процессе до импорта тяжёлых библиотек.
if sys.platform != "win32":
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (8, 8))
    resource.setrlimit(resource.RLIMIT_FSIZE, (6_000_000, 6_000_000))

import numpy as np
import pandas as pd

MAX_OUTPUT = 65_536
MAX_RESULT = 5 * 1024 * 1024
SAFE_BUILTINS = {
    "print": print,
    "len": len,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "sorted": sorted,
    "range": range,
    "int": int,
    "float": float,
    "str": str,
    "list": list,
    "dict": dict,
    "enumerate": enumerate,
    "zip": zip,
}


class BoundedOutput(io.StringIO):
    def write(self, text):
        if self.tell() + len(text) > MAX_OUTPUT:
            raise ValueError("Диагностический вывод превышает 64 KiB")
        return super().write(text)


def main():
    frame = pd.read_feather(sys.argv[1])
    namespace = {"__builtins__": SAFE_BUILTINS, "df": frame, "pd": pd, "np": np}
    output = BoundedOutput()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        exec(compile(sys.stdin.read(3501), "<analysis>", "exec"), namespace, namespace)
        value = namespace.get("result")
        result = None
        if isinstance(value, pd.Series):
            value = value.reset_index(name=value.name or "value")
        if isinstance(value, pd.DataFrame):
            if len(value) > 20_000 or len(value.columns) > 80:
                raise ValueError("Результат превышает 20 000 строк / 80 колонок")
            if not value.columns.is_unique or not all(isinstance(c, str) for c in value.columns):
                raise ValueError("В результате нужны уникальные строковые имена колонок")
            payload = value.to_json(orient="table", date_format="iso", index=False)
            if len(payload.encode()) > MAX_RESULT:
                raise ValueError("Результат превышает 5 MiB")
            result = {"kind": "table", "table": json.loads(payload)}
        elif value is not None:
            if isinstance(value, np.generic):
                value = value.item()
            if not isinstance(value, (int, float, bool, str)):
                raise ValueError("result должен быть числом, строкой, Series или DataFrame")
            if isinstance(value, float) and not np.isfinite(value):
                value = None
            result = {"kind": "scalar", "value": value}
    response = {"ok": True, "stdout": output.getvalue(), "result": result}
    encoded = json.dumps(response, ensure_ascii=False, allow_nan=False)
    if len(encoded.encode()) > MAX_RESULT + MAX_OUTPUT:
        raise ValueError("Слишком большой ответ worker")
    print(encoded)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
