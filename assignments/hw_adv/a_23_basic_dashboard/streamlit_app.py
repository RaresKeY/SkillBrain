import streamlit as st

from app_utils import get_data_source, load_employees_dataframe

st.set_page_config(page_title="Session 23", page_icon="📊", layout="wide")

st.title("Session 23: Basic Dashboard")

source = get_data_source()
if source:
    st.caption(f"Data source: {source.name}")
else:
    st.caption("Data source: fallback sample dataset")

df = load_employees_dataframe()

st.subheader("Raw Data")
st.dataframe(df, width="stretch")

if "department" in df.columns:
    departments = sorted(df["department"].dropna().astype(str).unique().tolist())
    selected_departments = st.multiselect("Filter by department", departments, default=departments)
else:
    selected_departments = []

min_salary = float(df["salary"].min()) if "salary" in df.columns else 0.0
max_salary = float(df["salary"].max()) if "salary" in df.columns else 0.0
salary_range = st.slider("Salary range", min_value=min_salary, max_value=max_salary, value=(min_salary, max_salary))

filtered = df.copy()
if selected_departments and "department" in filtered.columns:
    filtered = filtered[filtered["department"].astype(str).isin(selected_departments)]
if "salary" in filtered.columns:
    filtered = filtered[(filtered["salary"] >= salary_range[0]) & (filtered["salary"] <= salary_range[1])]

st.subheader("Filtered Data")
st.dataframe(filtered, width="stretch")

col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{len(filtered):,}")
col2.metric("Avg Salary", f"{filtered['salary'].mean():,.2f}" if len(filtered) else "0.00")
col3.metric("Max Salary", f"{filtered['salary'].max():,.2f}" if len(filtered) else "0.00")
