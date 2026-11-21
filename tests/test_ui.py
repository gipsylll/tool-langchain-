from io import BytesIO
from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from data_analyst.workspace import Workspace


def test_ui_queue_answer_and_navigation(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYST_DATA_DIR", str(tmp_path))
    st.cache_resource.clear()
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.session_state["codex_manual"] = True
    app.run(timeout=20)
    assert not app.exception
    app.text_area[0].input("Посчитай общую выручку")
    app.button(key="submit_question").click().run()
    assert not app.exception
    workspace = Workspace(tmp_path)
    run = workspace.runs(pending=True)[0]
    result = workspace.call(run["id"], "execute_python", {"code": "result = df['revenue'].sum()"})
    assert result["ok"]
    workspace.finish(run["id"], f"Выручка: {result['value']}", [result["result_id"]])
    app.run()
    assert not app.exception
    assert any("Выручка:" in item.value for item in app.markdown)
    for page in ("Обзор", "Данные", "Настройки"):
        next(r for r in app.radio if r.label == "Раздел").set_value(page).run()
        assert not app.exception
    st.cache_resource.clear()


def test_upload_and_demo_switch_selected_table(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYST_DATA_DIR", str(tmp_path))
    upload = BytesIO(b"name,amount\nExample,42\n")
    upload.name = "uploaded.csv"
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    st.cache_resource.clear()
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.run(timeout=20)
    assert not app.exception
    demo_session = app.session_state["selected_session"]

    next(b for b in app.button if b.label == "Открыть таблицу").click().run()
    assert not app.exception
    workspace = Workspace(tmp_path)
    uploaded_session = app.session_state["selected_session"]
    assert uploaded_session != demo_session
    assert workspace.session(uploaded_session)["name"] == "uploaded.csv"
    assert workspace.frame(uploaded_session)["amount"].tolist() == [42]

    next(b for b in app.button if b.label == "Демо: продажи").click().run()
    assert not app.exception
    assert app.session_state["selected_session"] == demo_session

    next(b for b in app.button if b.label == "Открыть таблицу").click().run()
    assert not app.exception
    assert app.session_state["selected_session"] == uploaded_session
    assert len(workspace.sessions()) == 2
    st.cache_resource.clear()


def test_codex_submit_starts_processing_without_chat(tmp_path, monkeypatch):
    from data_analyst import codex_backend

    monkeypatch.setenv("ANALYST_DATA_DIR", str(tmp_path))
    calls = []

    def start(workspace, run_id, effort, model):
        calls.append((run_id, effort, model))
        assert workspace.claim(run_id)
        workspace.set_progress(run_id, "planning", "Codex готовит расчёт", effort=effort)
        return True

    monkeypatch.setattr(codex_backend, "start_codex", start)
    st.cache_resource.clear()
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.run(timeout=20)
    assert app.button(key="submit_question").label == "Запустить Codex"
    app.text_area[0].input("Посчитай выручку")
    app.button(key="submit_question").click().run()
    assert not app.exception
    assert len(calls) == 1 and calls[0][1:] == ("low", "gpt-6-luna")
    assert any("Codex готовит расчёт" in item.value for item in app.info)
    app.run()
    assert len(calls) == 1
    st.cache_resource.clear()


def test_failed_codex_can_be_retried_without_losing_original(tmp_path, monkeypatch):
    from data_analyst import codex_backend

    monkeypatch.setenv("ANALYST_DATA_DIR", str(tmp_path))
    workspace = Workspace(tmp_path)
    session = workspace.add_dataset(b"channel,revenue\nsite,10\n", "sales.csv")
    failed = workspace.submit(session, "Доли каналов продаж", "codex")
    workspace.fail(failed, "Истёк таймаут")
    calls = []

    def start(current, run_id, effort, model):
        calls.append((run_id, effort, model))
        current.claim(run_id)
        current.set_progress(run_id, "waiting_model", "Ожидаем ответ")
        return True

    monkeypatch.setattr(codex_backend, "start_codex", start)
    st.cache_resource.clear()
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.run(timeout=20)
    app.button(key=f"retry_{failed}").click().run()
    assert not app.exception
    assert len(calls) == 1
    retried = workspace.run(calls[0][0])
    assert retried["id"] != failed and retried["status"] == "running"
    assert retried["question"] == "Доли каналов продаж"
    assert workspace.run(failed)["status"] == "failed"
    assert len(workspace.runs()) == 2
    st.cache_resource.clear()
