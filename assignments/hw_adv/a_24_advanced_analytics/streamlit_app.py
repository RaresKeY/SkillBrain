import streamlit as st

from app_utils import load_employees_dataframe

st.set_page_config(page_title="Session 24", page_icon="📈", layout="wide")

st.title("Session 24: Advanced Interactive Analytics")

df = load_employees_dataframe().copy()

numeric_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in df.columns if c not in numeric_cols]

if not numeric_cols:
    st.warning("No numeric columns available for analysis.")
    st.stop()

left, right = st.columns(2)
with left:
    x_col = st.selectbox("X axis", numeric_cols, index=0)
with right:
    y_col = st.selectbox("Y axis", numeric_cols, index=min(1, len(numeric_cols) - 1))

color_col = st.selectbox("Color by", ["(none)"] + cat_cols, index=0)

plot_df = df[[x_col, y_col] + ([] if color_col == "(none)" else [color_col])].dropna()

st.subheader("Scatter Plot")
st.scatter_chart(plot_df, x=x_col, y=y_col, color=None if color_col == "(none)" else color_col)

st.subheader("Correlation Matrix")
corr = df[numeric_cols].corr(numeric_only=True)
st.dataframe(corr.style.background_gradient(cmap="Blues"), width="stretch")

st.subheader("Grouped Summary")
if cat_cols:
    group_col = st.selectbox("Group by", cat_cols, index=0)
    agg_col = st.selectbox("Aggregate numeric column", numeric_cols, index=0)
    grouped = (
        df.groupby(group_col, dropna=False)[agg_col]
        .agg(["count", "mean", "min", "max"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    st.dataframe(grouped, width="stretch")
else:
    st.info("No categorical columns found for grouping.")
