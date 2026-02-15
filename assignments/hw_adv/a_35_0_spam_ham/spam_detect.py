import argparse
import csv
import re
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "spam.csv"
DEFAULT_MODEL_DIR = SCRIPT_DIR / "artifacts"
UCI_SMS_SPAM_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def try_download_uci_dataset(target_csv: Path) -> bool:
    try:
        print(f"[INFO] Downloading dataset from: {UCI_SMS_SPAM_URL}")
        raw = urlopen(UCI_SMS_SPAM_URL, timeout=30).read()
        with zipfile.ZipFile(BytesIO(raw)) as zf:
            with zf.open("SMSSpamCollection") as src:
                rows = []
                for line in src.read().decode("utf-8", errors="ignore").splitlines():
                    parts = line.split("\t", 1)
                    if len(parts) != 2:
                        continue
                    rows.append((parts[0], parts[1]))
        target_csv.parent.mkdir(parents=True, exist_ok=True)
        with target_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "text"])
            writer.writerows(rows)
        print(f"[INFO] Saved downloaded dataset to: {target_csv}")
        return True
    except Exception as exc:
        print(f"[WARN] Could not download UCI dataset: {exc}")
        return False


def load_dataset(dataset_path: Path, allow_download: bool = True) -> pd.DataFrame:
    if not dataset_path.exists() and allow_download:
        try_download_uci_dataset(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. Put a spam dataset there "
            f"with columns label/text, or run with --download."
        )

    df = pd.read_csv(dataset_path, encoding="latin-1")

    # Common formats:
    # 1) label,text
    # 2) v1,v2 (Kaggle SMS spam format)
    if {"label", "text"}.issubset(df.columns):
        use = df[["label", "text"]].copy()
    elif {"v1", "v2"}.issubset(df.columns):
        use = df[["v1", "v2"]].copy()
        use.columns = ["label", "text"]
    else:
        raise ValueError(
            "Unsupported dataset format. Expected columns: label/text or v1/v2."
        )

    use["label"] = use["label"].astype(str).str.lower().str.strip()
    use = use[use["label"].isin(["ham", "spam"])].copy()
    use["label_num"] = use["label"].map({"ham": 0, "spam": 1})
    use["text"] = use["text"].astype(str).map(clean_text)
    use = use[use["text"].str.len() > 0].reset_index(drop=True)

    if use.empty:
        raise ValueError("Dataset has no valid rows after preprocessing.")

    return use


def train_and_eval(
    X_train,
    X_test,
    y_train,
    y_test,
    vectorizer,
    model_name: str,
    artifact_path: Path,
):
    pipeline = Pipeline(
        [
            ("vectorizer", vectorizer),
            ("classifier", MultinomialNB()),
        ]
    )

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, target_names=["ham", "spam"])

    print(f"\n===== {model_name} =====")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print("Classification Report:")
    print(report)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact_path)
    print(f"[INFO] Saved model: {artifact_path}")

    return acc, pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Spam/Ham detector training script")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true", help="Download UCI SMS spam dataset if missing")
    parser.add_argument(
        "--predict",
        type=str,
        default="",
        help="Optional: run one prediction after training using best model",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    df = load_dataset(dataset_path, allow_download=args.download)

    print(f"[INFO] Rows: {len(df)}")
    print(f"[INFO] Class counts:\n{df['label'].value_counts()}")

    X = df["text"]
    y = df["label_num"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    count_acc, count_model = train_and_eval(
        X_train,
        X_test,
        y_train,
        y_test,
        CountVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english"),
        "CountVectorizer + MultinomialNB",
        DEFAULT_MODEL_DIR / "spam_ham_count_nb.joblib",
    )

    tfidf_acc, tfidf_model = train_and_eval(
        X_train,
        X_test,
        y_train,
        y_test,
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english", sublinear_tf=True),
        "TF-IDF + MultinomialNB",
        DEFAULT_MODEL_DIR / "spam_ham_tfidf_nb.joblib",
    )

    best_name = "tfidf" if tfidf_acc >= count_acc else "count"
    best_model = tfidf_model if tfidf_acc >= count_acc else count_model
    print(f"\n[INFO] Best baseline: {best_name} (accuracy={max(count_acc, tfidf_acc):.4f})")

    if args.predict:
        p = best_model.predict([clean_text(args.predict)])[0]
        label = "spam" if int(p) == 1 else "ham"
        print(f"[PREDICT] '{args.predict}' -> {label}")


if __name__ == "__main__":
    main()
