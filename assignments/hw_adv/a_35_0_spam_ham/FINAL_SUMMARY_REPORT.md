# Final Summary Report: Spam/Ham Project

## Project Goal
Build and compare two practical spam/ham text-classification approaches on SMS-style data:
- **Baseline:** TF-IDF + Multinomial Naive Bayes
- **Embedding-based:** Static embeddings v2 + Logistic Regression

Model repo used for embeddings:
- `LogicLark-QuantumQuill/static-embeddings-en-50m-v2`

## What Was Implemented
- Baseline trainer/evaluator: `spam_detect.py`
- Static-embedding trainer: `train_spam_ham_static_v2.py`
- Static-embedding tester/inference: `test_spam_ham_static_v2.py`
- Automated comparison runner: `run_automated_comparison_tests.py`
- Human-readable generated report: `automated_test_report.md`
- Custom evaluation set: `custom_test_dataset.csv`

## Important Technical Finding
Initial static-embedding performance was lower because the `.safetensors` file contained only the embedding matrix (no token mapping).  
After adding the provided vocab file:
- `static_embeddings_en_50m_pruned_fp16_v2.vocab.pth`

the pipeline switched to true token->index lookup and performance improved substantially.

## Datasets Used
- Real dataset: `spam.csv` (5,574 rows)
- Custom dataset: `custom_test_dataset.csv` (24 rows)

## Final Performance (Latest Automated Run)

### Real Holdout (from `spam.csv`)
- **TF-IDF + NB**
  - Accuracy: **97.49%**
  - Spam F1: **89.63%**
- **Static v2 + LogReg**
  - Accuracy: **95.87%**
  - Spam F1: **84.35%**

Result: baseline TF-IDF remains best on the real holdout split.

### Custom Dataset
- **TF-IDF + NB**
  - Accuracy: **70.83%**
  - Spam F1: **58.82%**
- **Static v2 + LogReg**
  - Accuracy: **87.50%**
  - Spam F1: **86.96%**

Result: static embeddings generalize better on this custom phrasing set.

## Interpretation
- **TF-IDF strengths:** excellent precision on familiar spam patterns in the original dataset.
- **Static embedding strengths:** better semantic generalization on varied wording in custom examples.
- Best practical setup depends on target traffic:
  - mostly in-domain SMS spam -> TF-IDF is very strong;
  - varied/new phrasing -> static embeddings can outperform.

## Artifacts
- Trained static model: `artifacts/spam_ham_static_v2_logreg.joblib`
- Static model metadata: `artifacts/spam_ham_static_v2_metadata.json`
- Full side-by-side report: `automated_test_report.md`

