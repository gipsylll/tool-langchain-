import hashlib
import io
import os
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from dotenv import load_dotenv

from data_analyst.agent import check_connection, run_gemini
from data_analyst.codex_backend import (
    DEFAULT_CODEX_EFFORT,
    DEFAULT_CODEX_MODEL,
    codex_available,
    resolve_model,
    start_codex,
)
from data_analyst.dataset import json_dump
from data_analyst.workspace import Workspace

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")
st.markdown(
    """<style>
.block-container {padding-top: 2rem; max-width: 1400px;}
h1 {letter-spacing: -.05em; font-weight: 750;}
[data-testid="stMetric"] {border: 1px solid #dce4ee; border-radius: 14px; padding: 16px;}
[data-testid="stSidebar"] {border-right: 1px solid #e3e8ef;}
[data-testid="stChatMessage"] {border-radius: 16px;}
</style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_workspace() -> Workspace:
    return Workspace()


workspace = get_workspace()


@st.cache_data(show_spinner=False, max_entries=8)
def get_frame(session_id: str) -> pd.DataFrame:
    return workspace.frame(session_id)


def load_file(contents: bytes, name: str) -> None:
    marker = hashlib.sha256(contents + name.encode()).hexdigest()
    loaded = st.session_state.setdefault("loaded_datasets", {})
    if marker not in loaded:
        loaded[marker] = workspace.add_dataset(contents, name)
    st.session_state.pending_session = loaded[marker]


sessions = workspace.sessions()
if not sessions and (ROOT / "data/demo_sales.csv").exists():
    load_file((ROOT / "data/demo_sales.csv").read_bytes(), "demo_sales.csv")
    sessions = workspace.sessions()

with st.sidebar:
    st.markdown("### AI Data Analyst")
    st.caption("Локальные расчёты · облачный интеллект")
    mode_label = st.radio("Как работаем", ["Через ChatGPT / Codex", "Gemini API"])
    mode = "codex" if mode_label == "Через ChatGPT / Codex" else "gemini"
    page = st.radio("Раздел", ["Анализ", "Обзор", "Данные", "Настройки"])
    st.divider()
    if sessions:
        ids = [s["id"] for s in sessions]
        names = {s["id"]: s["name"] for s in sessions}
        labels = {session: f"{name} · {session[:6]}" for session, name in names.items()}
        pending_session = st.session_state.pop("pending_session", None)
        if pending_session in ids:
            st.session_state.selected_session = pending_session
        elif st.session_state.get("selected_session") not in ids:
            st.session_state.selected_session = ids[0]
        session_id = st.selectbox(
            "Таблица", ids, format_func=lambda item: labels[item], key="selected_session"
        )
    else:
        session_id = None
    with st.expander("Загрузить данные"):
        upload = st.file_uploader("CSV / XLSX", type=["csv", "xlsx"])
        if st.button("Открыть таблицу", disabled=upload is None, width="stretch") and upload:
            try:
                load_file(upload.getvalue(), upload.name)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if st.button("Демо: продажи", width="stretch"):
            load_file((ROOT / "data/demo_sales.csv").read_bytes(), "demo_sales.csv")
            st.rerun()
    st.caption("Данные и результаты сохраняются на этом компьютере.")
    if mode == "codex":
        if st.session_state.get("codex_manual", False):
            st.info(
                "Ручной режим: сохраните вопрос, затем попросите ассистента в чате обработать его."
            )
        else:
            st.caption(
                "Codex запускается автоматически по кнопке. "
                f"Режим: {st.session_state.get('codex_effort', DEFAULT_CODEX_EFFORT)}."
            )
    else:
        st.caption(
            "Gemini получает вопрос, схему, примеры и результаты инструментов через API Google."
        )

st.title("Рабочее пространство аналитика")
st.caption("Вопрос → вычисления → таблица и график → вывод с источниками")

if page == "Настройки":
    st.subheader("Два способа работы")
    st.markdown(
        "**Codex:** приложение сразу запускает облачный Codex CLI с вашей авторизацией ChatGPT. "
        "Это отдельная сессия, а не обращение в текущий чат. Модель готовит план, "
        "приложение локально выполняет расчёт и строит график."
    )
    with st.form("codex_settings"):
        codex_model = st.text_input(
            "Codex: модель (пусто — Luna 6)",
            value=st.session_state.get("codex_model", DEFAULT_CODEX_MODEL),
            help="Укажите полный ID модели. Короткое luna сопоставляется с доступной моделью CLI.",
        )
        efforts = ["low", "medium", "high"]
        codex_effort = st.selectbox(
            "Codex: глубина анализа",
            efforts,
            index=efforts.index(st.session_state.get("codex_effort", DEFAULT_CODEX_EFFORT)),
            help="low — быстрый вариант; medium / high могут дольше готовить расчёт.",
        )
        codex_manual = st.checkbox(
            "Обрабатывать вручную через этот чат",
            value=st.session_state.get("codex_manual", False),
        )
        save_codex = st.form_submit_button("Применить настройки Codex")
    if save_codex:
        try:
            st.session_state.codex_model = resolve_model(codex_model.strip())
            st.session_state.codex_effort = codex_effort
            st.session_state.codex_manual = codex_manual
            st.success("Настройки Codex применены.")
        except ValueError as exc:
            st.error(str(exc))
    st.caption(
        "Codex CLI найден." if codex_available() else "Codex CLI не найден: нужен codex login."
    )
    st.markdown(
        "**Gemini API:** самостоятельный агент в приложении. Требуется ключ Google "
        "и доступная ему модель с Tool Calling."
    )
    with st.form("gemini_settings"):
        model_name = st.text_input(
            "Gemini: имя модели",
            value=st.session_state.get(
                "gemini_model", os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            ),
        )
        api_key = st.text_input(
            "Gemini API-ключ",
            type="password",
            value=st.session_state.get("gemini_key", os.getenv("GEMINI_API_KEY", "")),
        )
        st.caption(
            "Ключ остаётся в памяти этой сессии; в историю, журнал и экспорт не записывается."
        )
        save = st.form_submit_button("Применить")
        check = st.form_submit_button("Применить и проверить Tool Calling")
    if save or check:
        st.session_state.gemini_model = model_name.strip()
        st.session_state.gemini_key = api_key.strip()
        if check:
            if not api_key.strip() or not model_name.strip():
                st.warning("Введите ключ и имя модели.")
            else:
                with st.spinner("Проверяем соединение..."):
                    st.info(check_connection(model_name, api_key))
        else:
            st.success("Настройки применены.")
    st.stop()

if session_id is None:
    st.info("Загрузите CSV / XLSX или откройте демонстрационные данные.")
    st.stop()

frame = get_frame(session_id)
st.caption(f"{names[session_id]} · {len(frame):,} строк · {len(frame.columns)} колонок")


def render_artifacts(run_id: str) -> None:
    for artifact in workspace.artifacts(run_id):
        payload = artifact["payload"]
        if artifact["kind"] == "chart":
            st.plotly_chart(
                pio.from_json(json_dump(payload["figure"])),
                width="stretch",
                key=f"chart_{artifact['id']}",
            )
            st.caption(f"График по результату {payload['source_result_id']}")
        elif artifact["kind"] == "table":
            result = pd.read_json(io.StringIO(json_dump(payload["table"])), orient="table")
            with st.expander(f"Таблица · {len(result)} строк · {artifact['id'][:8]}"):
                st.dataframe(result, hide_index=True, width="stretch")
                st.download_button(
                    "Скачать CSV",
                    result.to_csv(index=False).encode("utf-8-sig"),
                    f"result_{artifact['id']}.csv",
                    key=f"csv_{artifact['id']}",
                )
        else:
            st.write(f"Вычислено: **{payload['value']}**")
            st.caption(f"Источник: {artifact['id']}")


if page == "Анализ":
    st.subheader("Что хотите узнать?")
    codex_manual = st.session_state.get("codex_manual", False)
    if mode == "codex" and not codex_manual:
        st.caption(
            "Запуск сразу из приложения · "
            f"{st.session_state.get('codex_model') or DEFAULT_CODEX_MODEL} · "
            f"{st.session_state.get('codex_effort', DEFAULT_CODEX_EFFORT)}"
        )
    examples = [
        "Посчитай средний чек по регионам и построй график.",
        "Сравни выручку сентября и октября по категориям. Отдели факты от гипотез.",
        "Покажи выручку по месяцам. Отметь пропущенные и неполные периоды.",
    ]
    with st.form("question"):
        question = st.text_area("Вопрос к таблице", placeholder=examples[0], height=90)
        st.caption("Например: " + examples[1])
        submitted = st.form_submit_button(
            ("Сохранить вопрос для чата" if codex_manual else "Запустить Codex")
            if mode == "codex"
            else "Запустить Gemini",
            type="primary",
            key="submit_question",
        )
    if submitted:
        key = st.session_state.get("gemini_key", os.getenv("GEMINI_API_KEY", ""))
        model = st.session_state.get("gemini_model", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
        if not question.strip():
            st.warning("Введите вопрос.")
        elif mode == "gemini" and not key:
            st.warning("Укажите Gemini API-ключ в разделе «Настройки».")
        else:
            run_id = workspace.submit(session_id, question, mode)
            if mode == "gemini":
                with st.spinner("Gemini выполняет анализ и вызывает локальные инструменты..."):
                    run_gemini(workspace, run_id, model, key)
            elif not codex_manual:
                if start_codex(
                    workspace,
                    run_id,
                    st.session_state.get("codex_effort", DEFAULT_CODEX_EFFORT),
                    st.session_state.get("codex_model", DEFAULT_CODEX_MODEL),
                ):
                    st.success("Обработка запущена. Текущий этап и результат появятся ниже.")
            else:
                st.success(
                    f"Вопрос {run_id[:8]} сохранён. "
                    "Чтобы выполнить анализ, отправьте ассистенту фразу из карточки ниже."
                )

    @st.fragment(run_every="1s" if mode == "codex" else None)
    def history() -> None:
        runs = workspace.runs(session_id)
        if not runs:
            st.info("Начните с вопроса. Ниже появятся ответ, графики и шаги анализа.")
        for run in reversed(runs):
            with st.container(border=True):
                st.markdown(f"**{run['question']}**")
                status = {
                    "pending": "Ожидает обработки из чата",
                    "running": "Выполняется",
                    "completed": "Готово",
                    "failed": "Остановлен",
                }[run["status"]]
                st.caption(f"{run['mode']} · {status} · {run['id'][:8]}")
                progress = workspace.progress(run["id"])
                if progress:
                    end = (
                        progress["updated"]
                        if run["status"] in {"completed", "failed"}
                        else time.time()
                    )
                    elapsed = max(0, end - progress["started"])
                    st.caption(f"{progress['model']} · {progress['effort']} · {elapsed:.0f} с")
                    if run["status"] in {"pending", "running"}:
                        if elapsed > 200:
                            workspace.fail(
                                run["id"], "Обработка прервана. Отправьте вопрос повторно."
                            )
                            st.warning("Обработка прервана. Отправьте вопрос повторно.")
                        else:
                            st.info(progress["message"])
                if run["status"] == "pending" and run["mode"] == "codex":
                    st.info(
                        "Анализ ещё не запущен. Отправьте эту фразу в чат ассистента, "
                        "у которого открыт проект:"
                    )
                    st.code(
                        f"Обработай запрос {run['id'][:8]} в локальной аналитике.", language=None
                    )
                if run["answer"]:
                    (st.error if run["status"] == "failed" else st.markdown)(run["answer"])
                if run["status"] == "failed" and run["mode"] == "codex" and not codex_manual:
                    if st.button("Повторить Codex", key=f"retry_{run['id']}"):
                        retry_id = workspace.submit(run["session_id"], run["question"], "codex")
                        start_codex(
                            workspace,
                            retry_id,
                            st.session_state.get("codex_effort", DEFAULT_CODEX_EFFORT),
                            st.session_state.get("codex_model", DEFAULT_CODEX_MODEL),
                        )
                        st.rerun()
                render_artifacts(run["id"])
                events = workspace.events(run["id"])
                with st.expander(f"Шаги анализа · {len(events)}"):
                    for event in events:
                        st.markdown(f"**{event['name']}** · {event['elapsed']:.2f} с")
                        st.json(
                            {"arguments": event["arguments"], "response": event["response"]},
                            expanded=False,
                        )
                if run["status"] in {"completed", "failed"}:
                    if st.button("Подготовить отчёт ZIP", key=f"prepare_{run['id']}"):
                        st.session_state[f"export_{run['id']}"] = workspace.export(run["id"])
                    if f"export_{run['id']}" in st.session_state:
                        st.download_button(
                            "Скачать отчёт, таблицы и графики",
                            st.session_state[f"export_{run['id']}"],
                            f"analysis_{run['id'][:8]}.zip",
                            mime="application/zip",
                            key=f"download_{run['id']}",
                        )

    history()
elif page == "Обзор":
    st.caption("Автоматический обзор данных. Это программные вычисления без вызова модели.")
    left, middle, right = st.columns(3)
    left.metric("Строк", f"{len(frame):,}")
    if "revenue" in frame and pd.api.types.is_numeric_dtype(frame["revenue"]):
        total = frame["revenue"].sum(min_count=1)
        middle.metric(
            "Выручка · единицы источника", f"{total:,.2f}" if pd.notna(total) else "Нет данных"
        )
        if "order_id" in frame:
            orders = frame["order_id"].nunique()
            right.metric(
                "Средний чек",
                f"{total / orders:,.2f}" if orders and pd.notna(total) else "Нет данных",
            )
            st.caption(
                "Средний чек = сумма revenue / число уникальных order_id. "
                "Строки без order_id требуют отдельной проверки."
            )
        if "order_date" in frame:
            temp = frame.assign(
                month=pd.to_datetime(frame["order_date"], errors="coerce").dt.to_period("M")
            )
            monthly = temp.groupby("month")["revenue"].sum(min_count=1)
            if not monthly.empty:
                monthly = monthly.reindex(
                    pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
                )
                plot = monthly.rename_axis("Месяц").reset_index(name="Выручка")
                plot["Месяц"] = plot["Месяц"].astype(str)
                st.plotly_chart(
                    px.line(plot, x="Месяц", y="Выручка", markers=True), width="stretch"
                )
                st.caption(
                    "Пробелы не заменяются нулём. "
                    "Полнота периодов по одной последней дате не определяется."
                )
    else:
        st.info(
            "Для финансовых метрик нужна явно указанная колонка revenue. "
            "Доступен анализ любой таблицы."
        )
elif page == "Данные":
    st.subheader("Схема и качество")
    st.dataframe(
        pd.DataFrame(
            {
                "Колонка": frame.columns,
                "Тип": frame.dtypes.astype(str).values,
                "Пропуски": frame.isna().sum().values,
                "Уникальных": frame.nunique().values,
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        f"Полных дубликатов: {frame.duplicated().sum()}. "
        "Исходные строки не исправляются автоматически."
    )
    st.subheader("Предпросмотр · первые 500 строк")
    st.dataframe(frame.head(500), width="stretch", hide_index=True)
