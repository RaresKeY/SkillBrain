import random
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_CANDIDATES = ["employees.csv", "angajati.csv", "employees.xlsx", "angajati.xlsx"]

NORMALIZATION_MAP = {
    "department": "department",
    "departament": "department",
    "dept": "department",
    "team": "department",
    "salary": "salary",
    "salariu": "salary",
    "income": "salary",
    "name": "name",
    "nume": "name",
    "employee": "name",
    "full_name": "name",
    "age": "age",
    "varsta": "age",
    "oras": "city",
    "city": "city",
    "nivel": "level",
    "level": "level",
    "seniority": "level",
    "years": "years_at_company",
    "tenure": "years_at_company",
    "years_at_company": "years_at_company",
    "performance": "performance_score",
    "performance_score": "performance_score",
}

DEPARTMENT_PROFILES = {
    "Engineering": {"weight": 17, "salary_band": (9000, 26000), "remote_mean": 72},
    "Data": {"weight": 9, "salary_band": (8800, 23500), "remote_mean": 68},
    "Product": {"weight": 8, "salary_band": (8500, 22000), "remote_mean": 65},
    "Finance": {"weight": 8, "salary_band": (7200, 18500), "remote_mean": 32},
    "HR": {"weight": 6, "salary_band": (6200, 14500), "remote_mean": 35},
    "Sales": {"weight": 14, "salary_band": (6500, 21000), "remote_mean": 38},
    "Marketing": {"weight": 10, "salary_band": (6800, 17000), "remote_mean": 48},
    "Operations": {"weight": 9, "salary_band": (6400, 16000), "remote_mean": 25},
    "Customer Success": {"weight": 8, "salary_band": (6200, 15500), "remote_mean": 45},
    "Legal": {"weight": 3, "salary_band": (9200, 22500), "remote_mean": 22},
    "Procurement": {"weight": 4, "salary_band": (6800, 15200), "remote_mean": 20},
    "R&D": {"weight": 4, "salary_band": (9300, 25000), "remote_mean": 76},
}

CITY_PROFILES = {
    "Bucharest": {"weight": 25, "salary_factor": 1.12},
    "Cluj-Napoca": {"weight": 15, "salary_factor": 1.08},
    "Timisoara": {"weight": 11, "salary_factor": 1.04},
    "Iasi": {"weight": 11, "salary_factor": 1.00},
    "Brasov": {"weight": 7, "salary_factor": 0.97},
    "Constanta": {"weight": 5, "salary_factor": 0.95},
    "Sibiu": {"weight": 6, "salary_factor": 0.96},
    "Oradea": {"weight": 6, "salary_factor": 0.94},
    "Craiova": {"weight": 5, "salary_factor": 0.93},
    "Pitesti": {"weight": 4, "salary_factor": 0.92},
    "Remote": {"weight": 5, "salary_factor": 0.99},
}

LEVEL_PROFILES = {
    "Junior": {"weight": 32, "salary_factor": 0.76, "age_range": (22, 31), "tenure_range": (0, 4)},
    "Mid": {"weight": 37, "salary_factor": 1.00, "age_range": (25, 40), "tenure_range": (2, 9)},
    "Senior": {"weight": 23, "salary_factor": 1.36, "age_range": (29, 50), "tenure_range": (4, 16)},
    "Lead": {"weight": 8, "salary_factor": 1.68, "age_range": (32, 57), "tenure_range": (6, 22)},
}

FIRST_NAMES = [
    "Ana",
    "Andrei",
    "Bianca",
    "Calin",
    "Daniela",
    "Elena",
    "Gabriel",
    "Ioana",
    "Laura",
    "Mihai",
    "Mirela",
    "Radu",
    "Stefan",
    "Teodora",
    "Vlad",
    "Alex",
    "Iulia",
    "Cristian",
    "Monica",
    "Denis",
]

LAST_NAMES = [
    "Ionescu",
    "Popescu",
    "Marin",
    "Dobre",
    "Georgescu",
    "Stoica",
    "Dumitru",
    "Vasilescu",
    "Ilie",
    "Enache",
    "Stan",
    "Voicu",
    "Matei",
    "Petrescu",
    "Luca",
]


def _normalize_column(col: str) -> str:
    key = str(col).strip().lower().replace(" ", "_").replace("-", "_")
    return NORMALIZATION_MAP.get(key, key)


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_normalize_column(c) for c in out.columns]
    return out


def _weighted_pick(rng: random.Random, options: list[str], weights: list[float]) -> str:
    return rng.choices(options, weights=weights, k=1)[0]


def _build_weighted_workforce(
    n_rows: int,
    seed: int = 24,
    extra_departments: list[str] | None = None,
    extra_cities: list[str] | None = None,
) -> pd.DataFrame:
    rng = random.Random(seed)

    departments = dict(DEPARTMENT_PROFILES)
    for dep in extra_departments or []:
        clean_dep = str(dep).strip()
        if clean_dep and clean_dep not in departments:
            departments[clean_dep] = {"weight": 2, "salary_band": (6500, 17000), "remote_mean": 45}

    cities = dict(CITY_PROFILES)
    for city in extra_cities or []:
        clean_city = str(city).strip()
        if clean_city and clean_city not in cities:
            cities[clean_city] = {"weight": 1, "salary_factor": 0.98}

    dep_names = list(departments.keys())
    dep_weights = [departments[name]["weight"] for name in dep_names]
    city_names = list(cities.keys())
    city_weights = [cities[name]["weight"] for name in city_names]
    level_names = list(LEVEL_PROFILES.keys())
    level_weights = [LEVEL_PROFILES[name]["weight"] for name in level_names]

    records = []
    for idx in range(1, n_rows + 1):
        department = _weighted_pick(rng, dep_names, dep_weights)
        dep_profile = departments[department]
        level = _weighted_pick(rng, level_names, level_weights)
        level_profile = LEVEL_PROFILES[level]
        city = _weighted_pick(rng, city_names, city_weights)
        city_profile = cities[city]

        age = rng.randint(*level_profile["age_range"])
        tenure = rng.randint(*level_profile["tenure_range"])
        tenure = min(tenure, max(0, age - 21))
        performance_score = round(max(2.2, min(5.0, rng.gauss(3.65, 0.55))), 2)

        salary_floor, salary_ceiling = dep_profile["salary_band"]
        base_salary = rng.uniform(salary_floor, salary_ceiling)
        salary = base_salary * level_profile["salary_factor"] * city_profile["salary_factor"]
        salary *= 1 + (tenure * 0.01)
        salary *= 1 + ((performance_score - 3.0) * 0.045)
        salary = round(max(4500, salary), 0)

        bonus_pct = max(0.0, min(0.30, rng.gauss(0.10 if level in {"Senior", "Lead"} else 0.06, 0.035)))
        bonus = round(salary * bonus_pct, 0)
        total_comp = round(salary + bonus, 0)
        remote_ratio = int(max(0, min(100, rng.gauss(dep_profile["remote_mean"], 18))))
        overtime_hours = int(max(0, rng.gauss(7 if department in {"Operations", "Sales"} else 4, 3)))
        training_hours = int(max(4, rng.gauss(26, 8)))
        fte = 1.0 if rng.random() > 0.08 else round(rng.uniform(0.6, 0.9), 2)

        records.append(
            {
                "employee_id": f"EMP{idx:05d}",
                "name": f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
                "department": department,
                "level": level,
                "city": city,
                "age": age,
                "years_at_company": tenure,
                "salary": salary,
                "bonus": bonus,
                "total_comp": total_comp,
                "performance_score": performance_score,
                "remote_ratio": remote_ratio,
                "overtime_hours": overtime_hours,
                "training_hours": training_hours,
                "fte": fte,
            }
        )

    return pd.DataFrame(records)


def _fallback_dataframe() -> pd.DataFrame:
    return _build_weighted_workforce(n_rows=1500, seed=24)


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
    if "city" not in out.columns:
        out["city"] = "Unknown"
    if "level" not in out.columns:
        out["level"] = "Mid"
    if "salary" not in out.columns:
        out["salary"] = 0.0
    if "name" not in out.columns:
        out["name"] = "Unknown"
    if "employee_id" not in out.columns:
        out["employee_id"] = [f"LEG{i:05d}" for i in range(1, len(out) + 1)]

    numeric_defaults = {
        "age": 30,
        "years_at_company": 3,
        "salary": 0.0,
        "bonus": 0.0,
        "total_comp": 0.0,
        "performance_score": 3.2,
        "remote_ratio": 40,
        "overtime_hours": 4,
        "training_hours": 20,
        "fte": 1.0,
    }
    for col, default_value in numeric_defaults.items():
        if col not in out.columns:
            out[col] = default_value
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default_value)

    out["total_comp"] = (out["salary"] + out["bonus"]).clip(lower=0)
    out["department"] = out["department"].astype(str).str.strip().replace({"": "Unknown"})
    out["city"] = out["city"].astype(str).str.strip().replace({"": "Unknown"})
    out["level"] = out["level"].astype(str).str.strip().replace({"": "Mid"})
    out["name"] = out["name"].astype(str).str.strip().replace({"": "Unknown"})
    return out


def _load_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def load_employees_dataframe(min_rows: int = 1500, seed: int = 24) -> pd.DataFrame:
    source = get_data_source()
    existing = pd.DataFrame()

    if source is not None:
        try:
            existing = _normalize_dataframe(_load_file(source))
            existing = ensure_columns(existing)
        except Exception:
            existing = pd.DataFrame()

    if existing.empty:
        return ensure_columns(_build_weighted_workforce(n_rows=min_rows, seed=seed))

    departments = sorted(existing["department"].dropna().astype(str).unique().tolist())
    cities = sorted(existing["city"].dropna().astype(str).unique().tolist()) if "city" in existing.columns else []
    target_rows = max(min_rows, len(existing))
    needed_rows = max(0, target_rows - len(existing))
    synthetic = _build_weighted_workforce(
        n_rows=needed_rows,
        seed=seed,
        extra_departments=departments,
        extra_cities=cities,
    )

    combined = pd.concat([existing, synthetic], ignore_index=True, sort=False)
    combined = ensure_columns(combined).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return combined.head(target_rows)
