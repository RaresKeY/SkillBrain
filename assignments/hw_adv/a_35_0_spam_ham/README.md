# Spam/Ham Detector

This folder contains a baseline spam vs ham text classifier.

## Files
- `spam_detect.py`: trains and evaluates CountVectorizer + TF-IDF baselines (Naive Bayes).
- `train_spam_ham_static_v2.py`: trains spam/ham classifier with your HF static embeddings v2.
- `test_spam_ham_static_v2.py`: evaluates/predicts using the trained v2-based classifier.
- `run_automated_comparison_tests.py`: automated side-by-side test runner for TF-IDF vs static-v2.
- `static_embeddings_utils.py`: shared loader/vectorization utilities for static embeddings.
- `custom_test_dataset.csv`: custom labeled spam/ham messages for focused testing.
- `SPAM_HAM_REPORT.md`: short notes and recommendations.
- `requirements.txt`: dependencies for this folder.

## Dataset
Default path:
- `assignments/hw_adv/a_35_0_spam_ham/spam.csv`

Supported CSV formats:
- `label,text`
- `v1,v2` (common SMS spam format)

If dataset is missing, you can attempt UCI download with `--download`.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r assignments/hw_adv/a_35_0_spam_ham/requirements.txt
```

## Train and Evaluate
```bash
python3 assignments/hw_adv/a_35_0_spam_ham/spam_detect.py --download
```

## Predict One Message
```bash
python3 assignments/hw_adv/a_35_0_spam_ham/spam_detect.py \
  --predict "Congratulations! You won a free prize"
```

## Output Artifacts
Saved models are written to:
- `assignments/hw_adv/a_35_0_spam_ham/artifacts/`

## Static Embeddings v2 Workflow
Model repo used:
- `LogicLark-QuantumQuill/static-embeddings-en-50m-v2`

Train:
```bash
python3 assignments/hw_adv/a_35_0_spam_ham/train_spam_ham_static_v2.py
```

Test on dataset:
```bash
python3 assignments/hw_adv/a_35_0_spam_ham/test_spam_ham_static_v2.py
```

Predict one message:
```bash
python3 assignments/hw_adv/a_35_0_spam_ham/test_spam_ham_static_v2.py \
  --text "win cash now call this number"
```

Notes:
- Script follows your v1 README usage pattern: static CBOW/Word2Vec embeddings as feature extractor.
- Document vectors are built via mean pooling of token vectors, then classified with Logistic Regression.

## Automated Comparison Tests + Markdown Report
Run:
```bash
python3 assignments/hw_adv/a_35_0_spam_ham/run_automated_comparison_tests.py
```

Output:
- `assignments/hw_adv/a_35_0_spam_ham/automated_test_report.md`

What it includes:
- Holdout metrics from the real dataset split.
- Metrics on `custom_test_dataset.csv`.
- Confusion matrices for both methods.
- Color-coded prediction examples (HAM/green, SPAM/red, PASS/FAIL).
