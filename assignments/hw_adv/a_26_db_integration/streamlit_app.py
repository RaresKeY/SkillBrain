import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from app_utils import BASE_DIR, load_employees_dataframe

st.set_page_config(page_title="Session 26", page_icon="🗃️", layout="wide")

st.title("Session 26: Streamlit + Database Integration")

DB_PATH = BASE_DIR / "employees.db"
TABLE_NAME = "employees"


def save_to_sqlite(df: pd.DataFrame) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)


def read_from_sqlite() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)


def table_exists(table_name: str) -> bool:
    if not DB_PATH.exists():
        return False
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table_name,),
        ).fetchone()
    return row is not None


st.subheader("Load Dataset")
mode = st.radio("Choose source", ["Local Excel", "Upload CSV", "Upload Excel"], horizontal=True)

if mode == "Local Excel":
    df = load_employees_dataframe()
elif mode == "Upload CSV":
    up = st.file_uploader("Upload CSV", type=["csv"])
    df = pd.read_csv(up) if up is not None else pd.DataFrame()
else:
    up = st.file_uploader("Upload Excel", type=["xlsx", "xls"])
    df = pd.read_excel(up) if up is not None else pd.DataFrame()

if df.empty:
    st.info("Load data to continue.")
    st.stop()

st.dataframe(df.head(20), width="stretch")

col1, col2 = st.columns(2)
if col1.button("Save To SQLite"):
    save_to_sqlite(df)
    st.success(f"Saved {len(df):,} rows to {DB_PATH.name}:{TABLE_NAME}")

if col2.button("Read From SQLite"):
    if not Path(DB_PATH).exists():
        st.warning("Database file does not exist yet. Save data first.")
    else:
        try:
            db_df = read_from_sqlite()
            st.dataframe(db_df, width="stretch")
        except Exception as exc:
            st.error(f"Could not read table: {exc}")

st.subheader("Query")
default_query = (
    f"SELECT * FROM {TABLE_NAME} LIMIT 20"
    if table_exists(TABLE_NAME)
    else "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
)
query = st.text_area("SQL query", value=default_query)
if st.button("Run Query"):
    if not DB_PATH.exists():
        st.warning("Database file does not exist yet. Save data first.")
        st.stop()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            out = pd.read_sql_query(query, conn)
        st.dataframe(out, width="stretch")
    except Exception as exc:
        st.error(f"Query failed: {exc}")
