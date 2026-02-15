# Spam/Ham Classifier Report

## What Was Built
A baseline spam/ham detector was implemented in `assignments/hw_adv/a_35_0_spam_ham/spam_detect.py`.

It trains and evaluates two models:
- `CountVectorizer + MultinomialNB`
- `TF-IDF + MultinomialNB`

It also:
- loads `spam.csv` in common formats (`label/text` or `v1/v2`),
- can download the UCI SMS spam dataset if `--download` is provided,
- prints accuracy, confusion matrix, and classification report,
- saves trained pipelines under `assignments/hw_adv/a_35_0_spam_ham/artifacts/`.

## CountVectorizer vs TF-IDF (Quick)
- `CountVectorizer`: raw term frequencies.
- `TF-IDF`: term frequencies weighted by how unique/informative terms are across documents.

For spam/ham, TF-IDF often performs slightly better because frequent generic words are downweighted.

## Can You Use Your Custom Word2Vec-Style Static Embeddings?
Short answer: **Yes, you can**.

Your models (`static-embeddings-en-50m-v1` and `static-embeddings-en-50m-v2`) are suitable as static token embeddings for a spam classifier, with a standard pipeline:
1. tokenize message,
2. map each token to embedding,
3. pool token vectors (mean/max/weighted mean),
4. train classifier (LogReg, SVM, MLP).

## Practical Considerations
- Static embeddings are faster and lighter than transformers.
- They can outperform TF-IDF when wording is varied but semantics are similar.
- They are still context-independent ("bank" has one vector), so they may miss nuanced intent.
- OOV handling is critical (unknown words, typos, SMS slang).

## v1 vs v2 Recommendation
- Prefer `v2` first for experiments (newer revision).
- Keep TF-IDF baseline as control.
- Pick final approach by validation metrics (F1 for spam class is especially important).

## Suggested Next Experiment
1. Keep current TF-IDF baseline metrics.
2. Add an embedding-based model using your v2 vectors.
3. Compare `spam` precision/recall/F1 and false-positive rate on ham.

## Example Run
```bash
python assignments/hw_adv/a_35_0_spam_ham/spam_detect.py --download
python assignments/hw_adv/a_35_0_spam_ham/spam_detect.py --predict "Congratulations! You won a free prize"
```
