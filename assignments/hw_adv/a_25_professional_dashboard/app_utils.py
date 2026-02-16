from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_CANDIDATES = ["employees.xlsx", "angajati.xlsx"]

NORMALIZATION_MAP = {
    "department": "department",
    "departament": "department",
    "dept": "department",
    "salary": "salary",
    "salariu": "salary",
    "income": "salary",
    "name": "name",
    "nume": "name",
    "employee": "name",
    "age": "age",
    "varsta": "age",
    "oras": "city",
    "city": "city",
}


def _normalize_column(col: str) -> str:
    key = str(col).strip().lower().replace(" ", "_")
    return NORMALIZATION_MAP.get(key, key)


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_normalize_column(c) for c in out.columns]
    return out


def _fallback_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"name": "Ana", "department": "IT", "salary": 11000, "age": 29, "city": "Bucharest"},
            {"name": "Mihai", "department": "Finance", "salary": 9800, "age": 34, "city": "Cluj"},
            {"name": "Ioana", "department": "HR", "salary": 7600, "age": 31, "city": "Iasi"},
            {"name": "Radu", "department": "Sales", "salary": 8700, "age": 28, "city": "Timisoara"},
            {"name": "Elena", "department": "IT", "salary": 12300, "age": 36, "city": "Bucharest"},
        ]
    )


def get_data_source() -> Path | None:
    for candidate in DATA_CANDIDATES:
        path = BASE_DIR / candidate
        if path.exists():
            return path
    return None


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "department" not in out.columns:
        out["department"] = "Unknown"
    if "salary" not in out.columns:
        out["salary"] = 0.0
    out["salary"] = pd.to_numeric(out["salary"], errors="coerce").fillna(0.0)
    return out


def load_employees_dataframe() -> pd.DataFrame:
    source = get_data_source()
    if source is None:
        return ensure_columns(_fallback_dataframe())
    try:
        df = pd.read_excel(source)
        df = _normalize_dataframe(df)
        return ensure_columns(df)
    except Exception:
        return ensure_columns(_fallback_dataframe())

