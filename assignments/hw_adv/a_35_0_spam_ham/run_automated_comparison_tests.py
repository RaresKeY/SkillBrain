import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from static_embeddings_utils import (
    DEFAULT_REPO_ID,
    download_and_load_embeddings,
    load_dataset,
    texts_to_avg_vectors,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REAL_DATASET = SCRIPT_DIR / "spam.csv"
DEFAULT_CUSTOM_DATASET = SCRIPT_DIR / "custom_test_dataset.csv"
DEFAULT_REPORT = SCRIPT_DIR / "automated_test_report.md"


def _to_label(y):
    return "spam" if int(y) == 1 else "ham"


def _label_color(label: str) -> str:
    return (
        '<span style="color:#b91c1c;font-weight:700;">SPAM</span>'
        if label == "spam"
        else '<span style="color:#166534;font-weight:700;">HAM</span>'
    )


def _status_color(ok: bool) -> str:
    return (
        '<span style="color:#166534;font-weight:700;">PASS</span>'
        if ok
        else '<span style="color:#b91c1c;font-weight:700;">FAIL</span>'
    )


def _metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_spam": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_spam": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_spam": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }


def _train_tfidf_nb(X_train, y_train):
    model = Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    stop_words="english",
                    sublinear_tf=True,
                ),
            ),
            ("classifier", MultinomialNB()),
        ]
    )
    model.fit(X_train, y_train)
    return model


def _train_static_v2(X_train, y_train, repo_id):
    kv, kv_path = download_and_load_embeddings(repo_id=repo_id, cache_dir=SCRIPT_DIR / "models")
    X_vec, train_cov = texts_to_avg_vectors(X_train, kv)
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_vec, y_train)
    return model, kv, kv_path, train_cov


def _eval_tfidf(model, texts, y):
    pred = model.predict(texts)
    return pred, _metrics(y, pred)


def _eval_static(model, kv, texts, y):
    X_vec, cov = texts_to_avg_vectors(texts, kv)
    pred = model.predict(X_vec)
    return pred, _metrics(y, pred), cov


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _metrics_table_row(name: str, m: dict) -> str:
    return (
        f"| {name} | {_fmt_pct(m['accuracy'])} | {_fmt_pct(m['precision_spam'])} | "
        f"{_fmt_pct(m['recall_spam'])} | {_fmt_pct(m['f1_spam'])} |"
    )


def _confusion_md(cm):
    tn, fp, fn, tp = cm.ravel()
    return "\n".join(
        [
            "|   | Pred HAM | Pred SPAM |",
            "|---|---:|---:|",
            f"| True HAM | {tn} | {fp} |",
            f"| True SPAM | {fn} | {tp} |",
        ]
    )


def _examples_md(df: pd.DataFrame, pred_a, pred_b, max_rows=16):
    rows = [
        "| # | Text | True | TF-IDF+NB | Static-v2+LogReg |",
        "|---:|---|---|---|---|",
    ]
    view = df.head(max_rows).copy()
    for idx, (_, row) in enumerate(view.iterrows(), start=1):
        text = str(row["text"]).replace("|", "\\|")
        true_label = _to_label(row["label_num"])
        a = _to_label(pred_a[idx - 1])
        b = _to_label(pred_b[idx - 1])

        a_ok = a == true_label
        b_ok = b == true_label

        rows.append(
            f"| {idx} | {text} | {_label_color(true_label)} | "
            f"{_label_color(a)} {_status_color(a_ok)} | "
            f"{_label_color(b)} {_status_color(b_ok)} |"
        )
    return "\n".join(rows)


def build_report(
    report_path: Path,
    repo_id: str,
    real_size: int,
    custom_size: int,
    tfidf_real: dict,
    static_real: dict,
    tfidf_custom: dict,
    static_custom: dict,
    tfidf_custom_pred,
    static_custom_pred,
    custom_df,
    kv_path: Path,
    train_cov: float,
    real_test_cov: float,
    custom_cov: float,
):
    winner_real = "TF-IDF + NB" if tfidf_real["f1_spam"] >= static_real["f1_spam"] else "Static-v2 + LogReg"
    winner_custom = "TF-IDF + NB" if tfidf_custom["f1_spam"] >= static_custom["f1_spam"] else "Static-v2 + LogReg"

    lines = [
        "# Automated Spam/Ham Comparison Report",
        "",
        f"Generated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Scope",
        f"- Real dataset: `{DEFAULT_REAL_DATASET}` ({real_size} rows)",
        f"- Custom dataset: `{DEFAULT_CUSTOM_DATASET}` ({custom_size} rows)",
        "- Compared methods:",
        "  - TF-IDF + MultinomialNB",
        "  - Static embeddings v2 + LogisticRegression",
        "",
        "## Model Data",
        f"- Static embeddings repo: `{repo_id}`",
        f"- Loaded embedding file: `{kv_path}`",
        f"- Static train token coverage: {_fmt_pct(train_cov)}",
        f"- Static holdout token coverage: {_fmt_pct(real_test_cov)}",
        f"- Static custom token coverage: {_fmt_pct(custom_cov)}",
        "",
        "## Holdout Test (From Real Dataset)",
        "| Method | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) |",
        "|---|---:|---:|---:|---:|",
        _metrics_table_row("TF-IDF + NB", tfidf_real),
        _metrics_table_row("Static-v2 + LogReg", static_real),
        "",
        f"Winner (F1 spam): **{winner_real}**",
        "",
        "### Confusion Matrix: TF-IDF + NB",
        _confusion_md(tfidf_real["confusion"]),
        "",
        "### Confusion Matrix: Static-v2 + LogReg",
        _confusion_md(static_real["confusion"]),
        "",
        "## Custom Dataset Test",
        "| Method | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) |",
        "|---|---:|---:|---:|---:|",
        _metrics_table_row("TF-IDF + NB", tfidf_custom),
        _metrics_table_row("Static-v2 + LogReg", static_custom),
        "",
        f"Winner (F1 spam): **{winner_custom}**",
        "",
        "### Confusion Matrix: TF-IDF + NB",
        _confusion_md(tfidf_custom["confusion"]),
        "",
        "### Confusion Matrix: Static-v2 + LogReg",
        _confusion_md(static_custom["confusion"]),
        "",
        "## Color-Coded Example Predictions (Custom Dataset)",
        "Legend: HAM/green, SPAM/red, PASS=correct, FAIL=wrong.",
        "",
        _examples_md(custom_df, tfidf_custom_pred, static_custom_pred),
        "",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated comparison tests: TF-IDF vs static embeddings v2")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_REAL_DATASET))
    parser.add_argument("--custom-dataset", type=str, default=str(DEFAULT_CUSTOM_DATASET))
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=str, default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    real_df = load_dataset(Path(args.dataset))
    custom_df = load_dataset(Path(args.custom_dataset))

    X_train, X_test, y_train, y_test = train_test_split(
        real_df["text"],
        real_df["label_num"],
        test_size=args.test_size,
        random_state=args.seed,
        stratify=real_df["label_num"],
    )

    tfidf_model = _train_tfidf_nb(X_train, y_train)
    tfidf_pred_real, tfidf_real_metrics = _eval_tfidf(tfidf_model, X_test, y_test)
    tfidf_pred_custom, tfidf_custom_metrics = _eval_tfidf(tfidf_model, custom_df["text"], custom_df["label_num"])

    static_model, kv, kv_path, train_cov = _train_static_v2(X_train, y_train, repo_id=args.repo_id)
    static_pred_real, static_real_metrics, real_cov = _eval_static(static_model, kv, X_test, y_test)
    static_pred_custom, static_custom_metrics, custom_cov = _eval_static(
        static_model,
        kv,
        custom_df["text"],
        custom_df["label_num"],
    )

    report_path = Path(args.report)
    build_report(
        report_path=report_path,
        repo_id=args.repo_id,
        real_size=len(real_df),
        custom_size=len(custom_df),
        tfidf_real=tfidf_real_metrics,
        static_real=static_real_metrics,
        tfidf_custom=tfidf_custom_metrics,
        static_custom=static_custom_metrics,
        tfidf_custom_pred=tfidf_pred_custom,
        static_custom_pred=static_pred_custom,
        custom_df=custom_df,
        kv_path=kv_path,
        train_cov=train_cov,
        real_test_cov=real_cov,
        custom_cov=custom_cov,
    )

    print(f"[INFO] Report written: {report_path}")


if __name__ == "__main__":
    main()
