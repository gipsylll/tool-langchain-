"""Проверяем реальный ChatOpenAI-адаптер Gemini без ключа и платного вызова."""

import json

import httpx
from langchain_core.messages import HumanMessage

from data_analyst.agent import AgentContext, build_graph, build_model
from data_analyst.workspace import Workspace


def test_gemini_compatible_http_tool_round_trip(tmp_path, monkeypatch):
    from data_analyst import gemini_model

    requests = []

    def respond(request):
        requests.append(request)
        body = json.loads(request.content)
        assert request.url.host == "generativelanguage.googleapis.com"
        assert request.url.path == "/v1beta/openai/chat/completions"
        if len(requests) == 1:
            name, arguments = "execute_python", {"code": "result = df['revenue'].sum()"}
        else:
            previous_call = next(
                message["tool_calls"][0]
                for message in body["messages"]
                if message.get("tool_calls")
            )
            assert previous_call["extra_content"]["google"]["thought_signature"] == "test-signature"
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
                                    "extra_content": {
                                        "google": {"thought_signature": "test-signature"}
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    original = gemini_model.GeminiChatOpenAI
    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(
        gemini_model, "GeminiChatOpenAI", lambda **kw: original(http_client=client, **kw)
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
    assert "test-signature" not in workspace.export(run_id).decode("latin1")
    client.close()


def test_gemini_error_logs_safe_status_without_request_or_response_data(
    tmp_path, monkeypatch, caplog
):
    import openai

    from data_analyst import agent

    secret = "fake-secret-for-test"
    private_data = "private-customer-data"
    request = httpx.Request("POST", "https://example.test", headers={"Authorization": secret})
    response = httpx.Response(
        400,
        request=request,
        json={"error": {"status": "INVALID_ARGUMENT", "message": f"{secret} {private_data}"}},
    )

    def fail(*args, **kwargs):
        raise openai.BadRequestError("sensitive body " + secret, response=response, body={})

    monkeypatch.setattr(agent, "build_model", fail)
    workspace = Workspace(tmp_path)
    session_id = workspace.add_dataset(b"revenue\n10\n", "data.csv")
    run_id = workspace.submit(session_id, private_data, "gemini")
    result = agent.run_gemini(workspace, run_id, "gemini-3.5-flash-lite", secret)

    assert result["status"] == "failed"
    assert "stage=analysis_failed" in caplog.text
    assert "http_status=400" in caplog.text
    assert "api_status=INVALID_ARGUMENT" in caplog.text
    assert secret not in caplog.text
    assert private_data not in caplog.text


def test_gemini_error_extracts_safe_hint_from_array_response(tmp_path, monkeypatch, caplog):
    import openai

    from data_analyst import agent

    request = httpx.Request("POST", "https://example.test")
    response = httpx.Response(
        400,
        request=request,
        json=[
            {
                "error": {
                    "status": "INVALID_ARGUMENT",
                    "message": "missing thought_signature secret-example",
                }
            }
        ],
    )

    def fail(*args, **kwargs):
        raise openai.BadRequestError("hidden", response=response, body={})

    monkeypatch.setattr(agent, "build_model", fail)
    workspace = Workspace(tmp_path)
    session_id = workspace.add_dataset(b"revenue\n10\n", "data.csv")
    run_id = workspace.submit(session_id, "test", "gemini")
    agent.run_gemini(workspace, run_id, "gemini-3.5-flash-lite", "fake-key")
    assert "api_status=INVALID_ARGUMENT" in caplog.text
    assert "error_hint=thought_signature" in caplog.text
    assert "secret-example" not in caplog.text


def test_openai_compatible_agent_runs_tool_loop(tmp_path, monkeypatch):
    import langchain_openai

    from data_analyst.agent import run_api_agent

    requests = []

    def respond(request):
        requests.append(request)
        body = json.loads(request.content)
        assert request.url.host == "api.example.test"
        assert body["model"] == "provider/model"
        if len(requests) == 1:
            name, arguments = "execute_python", {"code": "result = df['revenue'].sum()"}
        else:
            result = json.loads(body["messages"][-1]["content"])
            name, arguments = (
                "publish_answer",
                {"answer": "Выручка 30", "result_ids": [result["result_id"]]},
            )
        return httpx.Response(
            200,
            json={
                "id": "completion",
                "object": "chat.completion",
                "created": 1,
                "model": "provider/model",
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
            },
        )

    original = langchain_openai.ChatOpenAI
    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(
        langchain_openai, "ChatOpenAI", lambda **kwargs: original(http_client=client, **kwargs)
    )
    workspace = Workspace(tmp_path)
    session_id = workspace.add_dataset(b"revenue\n10\n20\n", "sales.csv")
    run_id = workspace.submit(session_id, "Сумма?", "api")
    result = run_api_agent(
        workspace,
        run_id,
        "provider/model",
        "fake-key",
        provider="OpenAI-compatible",
        base_url="https://api.example.test/v1",
    )
    assert result["status"] == "completed"
    assert len(requests) == 2
    assert workspace.artifacts(run_id)[0]["payload"]["value"] == 30
    assert "fake-key" not in workspace.export(run_id).decode("latin1")
    client.close()


def test_openai_compatible_rejects_unsafe_url():
    import pytest

    from data_analyst.agent import build_model

    for url in (
        "",
        "http://example.test/v1",
        "https://user:password@example.test/v1",
        "https://example.test/v1?key=secret",
    ):
        with pytest.raises(ValueError, match="HTTPS"):
            build_model("OpenAI-compatible", "model", "key", url)
