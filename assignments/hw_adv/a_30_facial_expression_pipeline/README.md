# Facial Expression Recognition Pipeline

This project is a complete pipeline for **synthetic data generation** and **facial expression recognition**. Here is how the components work together:

## 1. Data Generation (`generate_faces_batch.py`)
This script uses the Gemini API to generate hyper-realistic portraits. It uses **Pydantic templates** to define attributes like age, ethnicity, and sex, and **Asyncio** to generate images in parallel. This allows you to create a diverse dataset of hundreds of specific facial expressions very quickly.

**Key Features:**
*   Async batch processing for speed.
*   Diverse attribute combinations (Age, Sex, Ethnicity).
*   Targeted emotion generation.

## 2. Feature Extraction (`extract_landmarks.py`)
It processes the generated images using **MediaPipe Face Mesh**. It extracts **478 3D landmarks** for every face.
*   It parses the **emotion tag** from the filename.
*   It saves everything into `dataset/landmarks.csv` (the file you are currently viewing).

## 3. Visualization (`view_landmarks_ursina.py`)
This is a diagnostic tool built with the **Ursina Engine**. It renders the CSV data as a **3D point cloud**. You can use the arrow keys to cycle through the dataset and verify that the landmarks (especially around the mouth and eyes) accurately represent the emotion labeled in the CSV.

## 4. Model Training (`train_model.py` and `train_resnet.py`)
This project supports two different approaches to emotion recognition:

1.  **Landmark-based (`train_model.py`):**
    *   Uses geometric relationships between 478 3D landmarks.
    *   Compares **Random Forest, SVM, and MLP (Neural Net)** using Grid Search to find the best performer.
    *   Extremely lightweight and fast.

2.  **Image-based (`train_resnet.py`):**
    *   Fine-tunes a pre-trained **ResNet-50** deep learning model.
    *   Sees the whole face (texture, skin wrinkles, etc.), not just coordinates.
    *   Requires more compute (GPU recommended) but has higher potential accuracy.

## 5. Active Learning (`active_learning_agent.py`)
This is a **Human-in-the-Loop** (or rather, **LLM-in-the-Loop**) system to improve the model with minimal effort.
*   It identifies images in `dataset/random_expressions/` that the current model is most uncertain about.
*   It sends only those "hard" cases to **Gemini 1.5 Flash** for labeling.
*   Labeled images are automatically moved to the training set to be included in the next training run.

---

# How to Test & Use

### Batch Testing on Local Images
To test the landmark model on a folder of images:
```bash
python test_model.py
```

### Real-Time Webcam Detection (Landmarks)
Fast, stable detection using the geometric landmark model:
```bash
python realtime_emotion_detection.py
```

### Real-Time Webcam Detection (ResNet-50)
Deep learning detection using the vision-based ResNet model:
```bash
python realtime_resnet.py
```

**Note:** Ensure you have run the corresponding training script before attempting real-time inference.

---

# Installation
```bash
pip install -r requirements.txt
```
