# Advanced Training Guide (2026 Edition)

This guide covers two powerful methodologies for training high-performance machine learning models with limited labeled data: **Self-Supervised Learning (SSL)** and **Active Learning**.

---

## 1. Self-Supervised Learning (SSL)

Self-Supervised Learning is a technique where the model learns to understand the underlying structure of data (images, text, audio) without explicit human labels. It generates its own training signals from the data itself.

### Core Concept
Instead of mapping `Input -> Label` (Supervised), the model solves a "pretext task" like:
*   **Masked Autoencoders (MAE):** "Here is an image with 75% of the pixels removed. Reconstruct the missing parts."
*   **Contrastive Learning (SimCLR, DINOv2):** "Here are two augmented versions of the same image (cropped, colored differently). Learn that they are the *same* object, while pushing away representations of different images."

### The Workflow
1.  **Pre-training (Unlabeled):** Train a massive "Foundation Model" (like ResNet or ViT) on a large, unlabeled dataset using a pretext task. The model learns high-level features like edges, textures, shapes, and object parts.
2.  **Downstream Task (Labeled):**
    *   **Feature Extraction:** Freeze the pre-trained model. Pass your small labeled dataset through it to get "embeddings" (vectors of numbers representing the image). Train a simple classifier (like Logistic Regression or a small MLP) on these embeddings.
    *   **Fine-Tuning:** Add a classification head to the pre-trained model and train the whole network (or just the later layers) on your small labeled dataset with a low learning rate.

### Implementation Steps for Your Project
1.  **Gather Data:** Collect 10,000+ unlabeled face images (e.g., from generated batches or public datasets like Flickr-Faces-HQ). You don't need emotion labels.
2.  **Pre-train:** Use a library like `LightlySSL` or `PyTorch Lightning` to train a ResNet-50 using SimCLR on this folder.
3.  **Fine-tune:** Load the saved "backbone" weights. Add a linear layer for your 14 emotions. Train on your specific 200 labeled images.

---

## 2. Active Learning

Active Learning is a strategy where the learning algorithm interacts with the user (or an oracle) to label new data points with the desired outputs. It prioritizes labeling the most "informative" data.

### Core Concept
Randomly labeling data is inefficient because many samples are redundant (e.g., 50 nearly identical "happy" faces). Active Learning finds the "edge cases"—images the model is currently confused about—and asks you to label *only* those.

### The Loop
1.  **Initial Train:** Train a model on a tiny initial labeled set (e.g., 20 images).
2.  **Query (Inference):** Run the model on a large pool of *unlabeled* data.
3.  **Select Strategy:** Pick the samples where the model is least certain. Common strategies:
    *   **Least Confidence:** Select samples where the highest probability score is low (e.g., 0.3 for Happy, 0.28 for Sad...).
    *   **Entropy Sampling:** Select samples with the highest entropy (chaos) in their prediction distribution.
    *   **Margin Sampling:** Select samples where the difference between the top two predicted classes is smallest.
4.  **Label:** The human (you) provides labels for these specific selected samples.
5.  **Update:** Add these newly labeled samples to the training set and retrain the model.
6.  **Repeat.**

### Benefits
*   **Efficiency:** You might reach 90% accuracy with only 200 labeled images, whereas random selection might require 2,000.
*   **Quality:** It forces the dataset to represent the "decision boundaries" of the problem, handling hard cases better.

---

## Summary Comparison

| Feature | Self-Supervised Learning (SSL) | Active Learning |
| :--- | :--- | :--- |
| **Goal** | Learn better features from unlabeled data | Label the most useful data efficiently |
| **Data Requirement** | Large unlabeled dataset | Small initial labeled set + Pool of unlabeled |
| **Human Effort** | Low (only for final fine-tuning) | Medium (iterative labeling required) |
| **Best For** | "I have tons of images but no labels" | "Labeling is expensive/slow, I want to do as little as possible" |
