import contextlib
import io
import json
import sys
from typing import Literal, TypedDict

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


class TableResult(TypedDict):
    kind: Literal["table"]
    table: dict


class ScalarResult(TypedDict):
    kind: Literal["scalar"]
    value: int | float | bool | str | None


class WorkerResponse(TypedDict):
    ok: Literal[True]
    stdout: str
    result: TableResult | ScalarResult | None


class BoundedOutput(io.StringIO):
    """Ограничивает суммарный UTF-8 вывод анализа."""

    def __init__(self) -> None:
        super().__init__()
        self._size = 0

    def write(self, text: str) -> int:
        size = len(text.encode("utf-8"))
        if self._size + size > MAX_OUTPUT:
            raise ValueError("Диагностический вывод превышает 64 KiB")
        written = super().write(text)
        self._size += size
        return written


def serialize_result(value: object) -> TableResult | ScalarResult | None:
    if isinstance(value, pd.Series):
        value = value.reset_index(name=value.name or "value")
    if isinstance(value, pd.DataFrame):
        if len(value) > 20_000 or len(value.columns) > 80:
            raise ValueError("Результат превышает 20 000 строк / 80 колонок")
        if not value.columns.is_unique or not all(isinstance(c, str) for c in value.columns):
            raise ValueError("В результате нужны уникальные строковые имена колонок")
        payload = value.to_json(orient="table", date_format="iso", index=False)
        if len(payload.encode("utf-8")) > MAX_RESULT:
            raise ValueError("Результат превышает 5 MiB")
        return {"kind": "table", "table": json.loads(payload)}
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, (int, float, bool, str)):
        raise ValueError("result должен быть числом, строкой, Series или DataFrame")
    if isinstance(value, float) and not np.isfinite(value):
        value = None
    return {"kind": "scalar", "value": value}


def main() -> None:
    frame = pd.read_feather(sys.argv[1])
    namespace = {"__builtins__": SAFE_BUILTINS, "df": frame, "pd": pd, "np": np}
    output = BoundedOutput()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        exec(compile(sys.stdin.read(3501), "<analysis>", "exec"), namespace, namespace)
        result = serialize_result(namespace.get("result"))

    response: WorkerResponse = {"ok": True, "stdout": output.getvalue(), "result": result}
    encoded = json.dumps(response, ensure_ascii=False, allow_nan=False)
    if len(encoded.encode("utf-8")) > MAX_RESULT + MAX_OUTPUT:
        raise ValueError("Слишком большой ответ worker")
    print(encoded)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
