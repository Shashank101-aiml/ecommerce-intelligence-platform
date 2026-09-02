"""Shared DB-access helper for all dashboard pages."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.db.connection import get_engine  # noqa: E402


@st.cache_data(ttl=300)
def query(sql: str) -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(sql, engine)
