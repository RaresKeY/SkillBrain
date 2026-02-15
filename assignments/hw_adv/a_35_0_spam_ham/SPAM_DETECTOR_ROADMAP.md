# Building a Spam Detector: From Word Counts to AI

This guide outlines how to evolve your text analysis skills from simple word frequencies (as seen in `a_34_p1`) to a robust Spam vs. Ham classifier using Machine Learning.

## 1. The Bridge: From `word_freq.py` to Machine Learning

In `assignments/hw_adv/a_34_p1/word_freq.py`, you essentially built a manual **Bag of Words (BoW)** model.
- You split text into words.
- You counted occurrences.

**Machine Learning libraries automate this:**
- **Sklearn's `CountVectorizer`** does exactly what your loop does, but for thousands of documents instantly.
- **Sklearn's `TfidfVectorizer`** goes a step further: it lowers the weight of common words (like "the", "is") and boosts unique, meaningful words.

---

## 2. The Roadmap

We will approach this in three phases of increasing complexity.

### Phase 1: The Baseline (Scikit-Learn)
*Goal: Get a working model running quickly using statistical frequency.*

1.  **Data**: You need a dataset labeled `spam` and `ham` (e.g., the [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)).
2.  **Preprocessing**: Clean text (lowercase, remove punctuation) - similar to your `word_freq` function.
3.  **Vectorization**: Convert text to numbers using `TfidfVectorizer`.
4.  **Model**: Train a **Naive Bayes** (`MultinomialNB`) or **Support Vector Machine (SVM)** classifier.
5.  **Evaluation**: Check Accuracy and Confusion Matrix.

### Phase 2: Semantic Understanding (Gensim)
*Goal: Understand that "cheap" and "inexpensive" are similar, even if the exact words differ.*

1.  **Word Embeddings**: Use `gensim` to train a **Word2Vec** model on your data (or load a pre-trained one).
2.  **Vectorization**: Instead of counting words, represent each email as the *average* of its word vectors.
3.  **Model**: Train a classifier (like Random Forest or Logistic Regression) on these dense vectors.

### Phase 3: State-of-the-Art (Transformers)
*Goal: Understand context (e.g., "bank of the river" vs "bank account").*

1.  **Library**: `transformers` (Hugging Face).
2.  **Model**: Download a small, pre-trained model like `distilbert-base-uncased`.
3.  **Process**: Fine-tune the model on your spam dataset.
4.  **Result**: Highest accuracy, but requires more compute power (GPU recommended).

---

## 3. Practical Implementation Guide

### A. Setup Environment
Ensure you have the libraries installed:
```bash
pip install scikit-learn pandas numpy gensim transformers torch
```

### B. The Code Structure (for `spam_detect.py`)

#### Step 1: Load Data
```python
import pandas as pd

# Assuming a CSV with 'text' and 'label' columns
df = pd.read_csv("spam_data.csv")
# Convert labels to numbers (Spam=1, Ham=0)
df['label_num'] = df.label.map({'ham':0, 'spam':1})
```

#### Step 2: Split Data
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df.text, 
    df.label_num, 
    test_size=0.2
)
```

#### Step 3: Pipeline (Vectorize + Classify)
This replaces your manual `word_freq` logic with a pipeline.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Create a pipeline that first turns text into count vectors, then classifies
model = Pipeline([
    ('vectorizer', CountVectorizer()),  # <--- This is your automated word_freq
    ('classifier', MultinomialNB())
])

model.fit(X_train, y_train)
```

#### Step 4: Predict
```python
prediction = model.predict(["Congratulations! You won a lottery."])
print(prediction) # Should output [1] (Spam)
```

## 4. Where to get Data?
For this assignment, you can download the **SMS Spam Collection Dataset** from Kaggle or UCI Machine Learning Repository.
- Save it as `spam.csv` in this folder.

## 5. Next Steps for You
1.  Implement the **Phase 1** code in `spam_detect.py`.
2.  Run it and record the accuracy.
3.  Try changing `CountVectorizer` to `TfidfVectorizer` and see if it improves.
