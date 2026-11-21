"""Проверяем реальный ChatOpenAI-адаптер Gemini без ключа и платного вызова."""

import json

import httpx
from langchain_core.messages import HumanMessage

from data_analyst.agent import AgentContext, build_graph, build_model
from data_analyst.workspace import Workspace


def test_gemini_compatible_http_tool_round_trip(tmp_path, monkeypatch):
    import langchain_openai

    requests = []

    def respond(request):
        requests.append(request)
        body = json.loads(request.content)
        assert request.url.host == "generativelanguage.googleapis.com"
        assert request.url.path == "/v1beta/openai/chat/completions"
        if len(requests) == 1:
            name, arguments = "execute_python", {"code": "result = df['revenue'].sum()"}
        else:
            result = json.loads(body["messages"][-1]["content"])
            name, arguments = (
                "publish_answer",
                {
                    "answer": "Выручка 30.",
                    "result_ids": [result["result_id"]],
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "mock-completion",
                "object": "chat.completion",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": f"call-{len(requests)}",
                                    "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(arguments)},
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    original = langchain_openai.ChatOpenAI
    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(
        langchain_openai, "ChatOpenAI", lambda **kw: original(http_client=client, **kw)
    )
    workspace = Workspace(tmp_path)
    session_id = workspace.add_dataset(b"revenue\n10\n20\n", "sales.csv")
    run_id = workspace.submit(session_id, "Выручка?", "gemini")
    model = build_model("Gemini", "test-model", "fake-secret")
    build_graph(model, AgentContext(workspace, run_id)).invoke(
        {
            "messages": [HumanMessage(content="Выручка?")],
        }
    )
    assert len(requests) == 2
    assert workspace.run(run_id)["status"] == "completed"
    assert workspace.artifacts(run_id)[0]["payload"]["value"] == 30
    assert "fake-secret" not in workspace.export(run_id).decode("latin1")
    client.close()
