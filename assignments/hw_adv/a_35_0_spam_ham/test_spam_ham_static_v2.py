import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from static_embeddings_utils import (
    DEFAULT_REPO_ID,
    download_and_load_embeddings,
    load_dataset,
    texts_to_avg_vectors,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "spam.csv"
DEFAULT_ARTIFACT_DIR = SCRIPT_DIR / "artifacts"
DEFAULT_MODEL = DEFAULT_ARTIFACT_DIR / "spam_ham_static_v2_logreg.joblib"
DEFAULT_META = DEFAULT_ARTIFACT_DIR / "spam_ham_static_v2_metadata.json"


def _predict_one(text: str, clf, kv) -> None:
    X, cov = texts_to_avg_vectors([text], kv)
    pred = int(clf.predict(X)[0])
    proba = clf.predict_proba(X)[0]
    label = "spam" if pred == 1 else "ham"
    print(f"[PREDICT] {text}")
    print(f"[PREDICT] label={label} prob_ham={proba[0]:.4f} prob_spam={proba[1]:.4f} coverage={cov:.2%}")


def _eval_dataset(dataset_path: Path, clf, kv) -> None:
    df = load_dataset(dataset_path)
    X, cov = texts_to_avg_vectors(df["text"], kv)
    y_true = df["label_num"]
    y_pred = clf.predict(X)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["ham", "spam"])

    print(f"[INFO] Eval rows: {len(df)}")
    print(f"[INFO] Token coverage: {cov:.2%}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print("Classification Report:")
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test/infer spam/ham classifier trained on static embeddings v2."
    )
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--meta-path", type=str, default=str(DEFAULT_META))
    parser.add_argument("--text", type=str, default="")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Classifier missing: {model_path}. Run train_spam_ham_static_v2.py first."
        )

    repo_id = args.repo_id
    meta_path = Path(args.meta_path)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        repo_id = meta.get("repo_id", repo_id)

    print(f"[INFO] Loading classifier: {model_path}")
    clf = joblib.load(model_path)
    print(f"[INFO] Loading embeddings from HF repo: {repo_id}")
    kv, kv_path = download_and_load_embeddings(repo_id=repo_id, cache_dir=SCRIPT_DIR / "models")
    print(f"[INFO] Loaded vectors from: {kv_path}")

    if args.text:
        _predict_one(args.text, clf, kv)
    else:
        _eval_dataset(Path(args.dataset), clf, kv)


if __name__ == "__main__":
    main()

