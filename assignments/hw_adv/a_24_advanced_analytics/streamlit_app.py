import streamlit as st

from app_utils import load_employees_dataframe

st.set_page_config(page_title="Session 24", page_icon="📈", layout="wide")

st.title("Session 24: Advanced Interactive Analytics")
st.caption("Larger weighted workforce dataset with multi-section analytics workflow.")

df = load_employees_dataframe().copy()
numeric_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in df.columns if c not in numeric_cols]

if df.empty or not numeric_cols:
    st.warning("No analyzable data available.")
    st.stop()

st.sidebar.header("Filters")

all_departments = sorted(df["department"].astype(str).dropna().unique().tolist()) if "department" in df.columns else []
all_cities = sorted(df["city"].astype(str).dropna().unique().tolist()) if "city" in df.columns else []
all_levels = sorted(df["level"].astype(str).dropna().unique().tolist()) if "level" in df.columns else []

selected_departments = (
    st.sidebar.multiselect("Departments", all_departments, default=all_departments) if all_departments else []
)
selected_cities = st.sidebar.multiselect("Cities", all_cities, default=all_cities) if all_cities else []
selected_levels = st.sidebar.multiselect("Levels", all_levels, default=all_levels) if all_levels else []

salary_min = int(df["salary"].min()) if "salary" in df.columns else 0
salary_max = int(df["salary"].max()) if "salary" in df.columns else 0
salary_range = st.sidebar.slider(
    "Salary range",
    min_value=salary_min,
    max_value=max(salary_min + 1, salary_max),
    value=(salary_min, max(salary_min + 1, salary_max)),
)

min_tenure = st.sidebar.slider("Min years at company", min_value=0, max_value=25, value=0)

filtered = df.copy()
if selected_departments:
    filtered = filtered[filtered["department"].astype(str).isin(selected_departments)]
if selected_cities:
    filtered = filtered[filtered["city"].astype(str).isin(selected_cities)]
if selected_levels:
    filtered = filtered[filtered["level"].astype(str).isin(selected_levels)]
filtered = filtered[filtered["salary"].between(salary_range[0], salary_range[1])]
if "years_at_company" in filtered.columns:
    filtered = filtered[filtered["years_at_company"] >= min_tenure]

if filtered.empty:
    st.info("No rows match the current filters. Expand filters in the sidebar.")
    st.stop()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Employees", f"{len(filtered):,}")
k2.metric("Avg Salary", f"{filtered['salary'].mean():,.0f}")
k3.metric("Median Salary", f"{filtered['salary'].median():,.0f}")
k4.metric("Avg Bonus", f"{filtered['bonus'].mean():,.0f}" if "bonus" in filtered.columns else "N/A")
k5.metric(
    "Avg Performance",
    f"{filtered['performance_score'].mean():.2f}" if "performance_score" in filtered.columns else "N/A",
)

overview_tab, explore_tab, corr_tab, segment_tab, data_tab = st.tabs(
    ["Overview", "Exploration", "Correlation", "Segment Table", "Data"]
)

with overview_tab:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Headcount by Department")
        headcount = (
            filtered.groupby("department", dropna=False)["employee_id"]
            .count()
            .sort_values(ascending=False)
            .rename("employees")
        )
        st.bar_chart(headcount)

        st.subheader("Salary Distribution")
        bins = filtered["salary"].value_counts(bins=18, sort=False)
        bins_df = bins.reset_index()
        bins_df.columns = ["salary_bin", "count"]
        bins_df["salary_bin"] = bins_df["salary_bin"].astype(str)
        st.bar_chart(bins_df, x="salary_bin", y="count")

    with col_b:
        st.subheader("Payroll by Department")
        payroll = filtered.groupby("department", dropna=False)["salary"].sum().sort_values(ascending=False)
        st.bar_chart(payroll)

        st.subheader("Top Cities by Average Salary")
        city_salary = (
            filtered.groupby("city", dropna=False)["salary"]
            .mean()
            .sort_values(ascending=False)
            .head(12)
            .rename("avg_salary")
        )
        st.bar_chart(city_salary)

with explore_tab:
    st.subheader("Scatter Explorer")
    left, right = st.columns(2)
    with left:
        x_col = st.selectbox("X axis", numeric_cols, index=numeric_cols.index("salary") if "salary" in numeric_cols else 0)
    with right:
        y_default = numeric_cols.index("age") if "age" in numeric_cols else min(1, len(numeric_cols) - 1)
        y_col = st.selectbox("Y axis", numeric_cols, index=y_default)

    color_options = ["(none)"] + [c for c in ["department", "city", "level"] if c in filtered.columns]
    color_col = st.selectbox("Color by", color_options, index=1 if len(color_options) > 1 else 0)
    slider_max = min(5000, len(filtered))
    slider_min = min(200, slider_max)
    sample_size = st.slider(
        "Max points",
        min_value=max(1, slider_min),
        max_value=max(1, slider_max),
        value=min(1800, max(1, slider_max)),
    )
    plot_df = filtered.sample(n=sample_size, random_state=24) if len(filtered) > sample_size else filtered
    cols = [x_col, y_col] + ([] if color_col == "(none)" else [color_col])
    plot_df = plot_df[cols].dropna()
    st.scatter_chart(plot_df, x=x_col, y=y_col, color=None if color_col == "(none)" else color_col)

with corr_tab:
    st.subheader("Correlation Matrix")
    default_cols = [c for c in ["salary", "bonus", "total_comp", "age", "years_at_company", "performance_score"] if c in numeric_cols]
    selected_numeric = st.multiselect(
        "Columns for correlation",
        numeric_cols,
        default=default_cols[:6] if default_cols else numeric_cols[:4],
    )
    if len(selected_numeric) < 2:
        st.info("Select at least two numeric columns.")
    else:
        corr = filtered[selected_numeric].corr(numeric_only=True)
        try:
            st.dataframe(corr.style.background_gradient(cmap="Blues"), width="stretch")
        except ImportError:
            st.warning("Install matplotlib to enable correlation heatmap styling.")
            st.dataframe(corr, width="stretch")
        if "salary" in corr.columns:
            st.subheader("Correlation with Salary")
            salary_corr = corr["salary"].drop("salary", errors="ignore").sort_values(ascending=False)
            st.bar_chart(salary_corr)

with segment_tab:
    st.subheader("Grouped Statistics")
    group_options = [c for c in ["department", "city", "level"] if c in filtered.columns] + [c for c in cat_cols if c not in {"department", "city", "level"}]
    metric_options = [c for c in ["salary", "bonus", "total_comp", "performance_score", "years_at_company"] if c in numeric_cols]
    if not group_options or not metric_options:
        st.info("Insufficient columns for grouped analytics.")
    else:
        group_col = st.selectbox("Group by", group_options, index=0)
        metric_col = st.selectbox("Metric", metric_options, index=0)
        grouped = (
            filtered.groupby(group_col, dropna=False)[metric_col]
            .agg(["count", "mean", "median", "min", "max"])
            .sort_values("mean", ascending=False)
            .reset_index()
        )
        grouped["mean"] = grouped["mean"].round(2)
        grouped["median"] = grouped["median"].round(2)
        st.dataframe(grouped, width="stretch")

with data_tab:
    st.subheader("Filtered Dataset")
    st.dataframe(filtered.sort_values("salary", ascending=False), width="stretch")
    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", csv_data, file_name="session24_filtered.csv", mime="text/csv")
