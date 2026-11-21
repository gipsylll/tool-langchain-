#!/bin/zsh
set -eu
cd "$(dirname "$0")"
if ! .venv/bin/python -c 'import streamlit, data_analyst' >/dev/null 2>&1; then
    uv sync
fi
exec .venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port "${ANALYST_PORT:-8502}"
