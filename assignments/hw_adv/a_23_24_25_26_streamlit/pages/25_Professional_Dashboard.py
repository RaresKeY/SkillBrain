import streamlit as st

from _bootstrap import ensure_project_root_on_path

st.set_page_config(page_title="Session 25", page_icon="📉", layout="wide")

ensure_project_root_on_path()

from app_utils import load_employees_dataframe

st.title("Session 25: Professional Dashboard")

df = load_employees_dataframe().copy()

st.sidebar.header("Controls")
if "department" in df.columns:
    departments = sorted(df["department"].dropna().astype(str).unique().tolist())
    selected = st.sidebar.multiselect("Departments", departments, default=departments)
    if selected:
        df = df[df["department"].astype(str).isin(selected)]

k1, k2, k3, k4 = st.columns(4)
rows = len(df)
avg_salary = float(df["salary"].mean()) if rows else 0.0
median_salary = float(df["salary"].median()) if rows else 0.0
salary_sum = float(df["salary"].sum()) if rows else 0.0

k1.metric("Employees", f"{rows:,}")
k2.metric("Avg Salary", f"{avg_salary:,.0f}")
k3.metric("Median Salary", f"{median_salary:,.0f}")
k4.metric("Total Payroll", f"{salary_sum:,.0f}")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Payroll by Department")
    if "department" in df.columns:
        by_dept = df.groupby("department", dropna=False)["salary"].sum().sort_values(ascending=False)
        st.bar_chart(by_dept)
    else:
        st.info("Missing 'department' column.")

with c2:
    st.subheader("Salary Distribution")
    binned = df["salary"].value_counts(bins=15, sort=False)
    binned_df = binned.reset_index()
    binned_df.columns = ["salary_bin", "count"]
    binned_df["salary_bin"] = binned_df["salary_bin"].astype(str)
    st.bar_chart(binned_df, x="salary_bin", y="count")

st.subheader("Data Table")
st.dataframe(df.sort_values("salary", ascending=False), width="stretch")
