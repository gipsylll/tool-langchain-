import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from data_analyst.workspace import Workspace


def test_stdio_client_executes_and_publishes(tmp_path):
    workspace = Workspace(tmp_path)
    session_id = workspace.add_dataset(b"revenue\n10\n20\n", "sales.csv")
    run_id = workspace.submit(session_id, "Посчитай", "codex")

    async def exercise():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "data_analyst.mcp_server"],
            env={**os.environ, "ANALYST_DATA_DIR": str(tmp_path)},
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                listing = await client.list_tools()
                assert {t.name for t in listing.tools} >= {
                    "pending_requests",
                    "execute_python",
                    "create_chart",
                    "publish_answer",
                }
                pending = await client.call_tool("pending_requests", {})
                assert run_id in pending.content[0].text
                response = await client.call_tool(
                    "execute_python",
                    {
                        "run_id": run_id,
                        "code": "result = df['revenue'].sum()",
                    },
                )
                result = json.loads(response.content[0].text)
                assert result["ok"] and result["value"] == 30
                done = await client.call_tool(
                    "publish_answer",
                    {
                        "run_id": run_id,
                        "answer": "Выручка: 30",
                        "result_ids": [result["result_id"]],
                    },
                )
                assert not done.isError

    asyncio.run(exercise())
    assert workspace.run(run_id)["answer"] == "Выручка: 30"
