import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from data_analyst.agent import AgentContext, build_graph, build_model, make_tools
from data_analyst.workspace import Workspace


class FakeToolCallingModel:
    def bind_tools(self, tools):
        self.names = [tool.name for tool in tools]
        return self

    def invoke(self, messages):
        if isinstance(messages[-1], HumanMessage):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute_python",
                        "args": {"code": "result = df['revenue'].sum()"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        response = json.loads(messages[-1].content)
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "publish_answer",
                    "args": {"answer": "Выручка 30.", "result_ids": [response["result_id"]]},
                    "id": "call_2",
                    "type": "tool_call",
                }
            ],
        )


def test_graph_computes_and_publishes_with_evidence(tmp_path):
    workspace = Workspace(tmp_path)
    session = workspace.add_dataset(b"order_id,revenue\n1,10\n2,20\n", "sales.csv")
    run_id = workspace.submit(session, "Посчитай выручку", "gemini")
    graph = build_graph(FakeToolCallingModel(), AgentContext(workspace, run_id))
    graph.invoke({"messages": [HumanMessage(content="Посчитай выручку")]})
    assert workspace.run(run_id)["answer"] == "Выручка 30."
    assert workspace.run(run_id)["status"] == "completed"
    assert workspace.artifacts(run_id)[0]["payload"]["value"] == 30


def test_model_without_computation_cannot_finish(tmp_path):
    class NoTools:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            return AIMessage(content="Выручка 9999.")

    workspace = Workspace(tmp_path)
    session = workspace.add_dataset(b"revenue\n10\n", "sales.csv")
    run_id = workspace.submit(session, "Посчитай", "gemini")
    graph = build_graph(NoTools(), AgentContext(workspace, run_id))
    with pytest.raises(ValueError, match="без publish_answer"):
        graph.invoke({"messages": [HumanMessage(content="Посчитай")]})


def test_gemini_uses_only_official_endpoint():
    with pytest.raises(ValueError, match="официальный"):
        build_model("Gemini", "gemini-example", "secret", "http://localhost.example.com")


def test_tool_schemas_describe_arguments(tmp_path):
    tools = {tool.name: tool for tool in make_tools(AgentContext(Workspace(tmp_path), "run"))}
    code = tools["execute_python"].args["code"]
    assert "result" in code["description"]
    chart_type = tools["create_chart"].args["chart_type"]
    assert set(chart_type["enum"]) == {"bar", "line", "area", "scatter"}
    assert "вычислений" in tools["publish_answer"].description
