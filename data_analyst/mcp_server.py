"""MCP stdio: внешняя модель использует те же результаты, что Gemini и UI."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from data_analyst.workspace import Workspace

mcp = FastMCP("Local Data Analyst")


def workspace():
    return Workspace()


@mcp.tool()
def list_sessions() -> list[dict]:
    """Найти загруженные в локальном приложении таблицы и их session_id."""
    return workspace().sessions()


@mcp.tool()
def pending_requests() -> list[dict]:
    """Получить вопросы пользователя из локального приложения для Codex / ChatGPT."""
    return workspace().runs(pending=True)


@mcp.tool()
def inspect_data(run_id: str) -> dict:
    """Исследовать схему и примеры таблицы текущего запроса."""
    return workspace().call(run_id, "inspect_data", {})


@mcp.tool()
def execute_python(run_id: str, code: str) -> dict:
    """Вычислить над df (pd/np доступны). Присвоить итог переменной result. Вернёт result_id."""
    return workspace().call(run_id, "execute_python", {"code": code})


@mcp.tool()
def create_chart(
    run_id: str,
    result_id: str,
    x: str,
    y: str,
    chart_type: str = "bar",
    title: str = "Результат анализа",
) -> dict:
    """Построить график по табличному result_id без повторной агрегации."""
    return workspace().call(
        run_id,
        "create_chart",
        {
            "result_id": result_id,
            "x": x,
            "y": y,
            "chart_type": chart_type,
            "title": title,
        },
    )


@mcp.tool()
def publish_answer(run_id: str, answer: str, result_ids: list[str]) -> dict:
    """Опубликовать ответ в локальном приложении с ID подтверждающих вычислений."""
    return workspace().finish(run_id, answer, result_ids)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
