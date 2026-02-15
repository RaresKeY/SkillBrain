import argparse
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from static_embeddings_utils import (
    DEFAULT_REPO_ID,
    download_and_load_embeddings,
    load_dataset,
    texts_to_avg_vectors,
    write_metadata,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "spam.csv"
DEFAULT_ARTIFACT_DIR = SCRIPT_DIR / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train spam/ham classifier using static embeddings v2 (CBOW/Word2Vec style)."
    )
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_dataset(Path(args.dataset))
    print(f"[INFO] Rows: {len(df)}")
    print(f"[INFO] Class counts:\n{df['label'].value_counts()}")

    print(f"[INFO] Loading embeddings from HF repo: {args.repo_id}")
    kv, kv_path = download_and_load_embeddings(repo_id=args.repo_id, cache_dir=SCRIPT_DIR / "models")
    print(f"[INFO] Loaded vectors from: {kv_path}")
    print(f"[INFO] Embedding dim: {kv.vector_size}")
    print(f"[INFO] Embedding vocab size: {len(kv.key_to_index)}")

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"],
        df["label_num"],
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["label_num"],
    )

    X_train, train_cov = texts_to_avg_vectors(X_train_text, kv)
    X_test, test_cov = texts_to_avg_vectors(X_test_text, kv)
    print(f"[INFO] Train token coverage: {train_cov:.2%}")
    print(f"[INFO] Test token coverage: {test_cov:.2%}")

    clf = LogisticRegression(max_iter=2000, random_state=args.seed)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, target_names=["ham", "spam"])

    print("\n===== Static Embeddings v2 + LogisticRegression =====")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print("Classification Report:")
    print(report)

    DEFAULT_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_out = DEFAULT_ARTIFACT_DIR / "spam_ham_static_v2_logreg.joblib"
    meta_out = DEFAULT_ARTIFACT_DIR / "spam_ham_static_v2_metadata.json"
    joblib.dump(clf, model_out)

    write_metadata(
        meta_out,
        {
            "repo_id": args.repo_id,
            "embedding_file": str(kv_path),
            "embedding_dim": int(kv.vector_size),
            "embedding_vocab_size": int(len(kv.key_to_index)),
            "test_size": float(args.test_size),
            "seed": int(args.seed),
            "accuracy": float(acc),
            "train_token_coverage": float(train_cov),
            "test_token_coverage": float(test_cov),
        },
    )
    print(f"[INFO] Saved classifier: {model_out}")
    print(f"[INFO] Saved metadata: {meta_out}")


if __name__ == "__main__":
    main()

