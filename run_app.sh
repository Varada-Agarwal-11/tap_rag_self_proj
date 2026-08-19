#!/usr/bin/env bash
set -e
python scripts/build_database.py
streamlit run app.py
