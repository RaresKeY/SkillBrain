import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# --- Configuration ---
BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "dataset" / "landmarks.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = MODELS_DIR / "emotion_model.pkl"
ENCODER_SAVE_PATH = MODELS_DIR / "label_encoder.pkl"
SCALER_SAVE_PATH = MODELS_DIR / "scaler.pkl"

def train():
    print("--- Loading Dataset ---")
    if not DATASET_PATH.exists():
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    df = pd.read_csv(DATASET_PATH)

    if len(df) < 20:
        print("Error: Dataset too small. Please generate more samples.")
        return

    # --- Filtering for Stratification ---
    # Stratified split requires at least 2 samples per class.
    counts = df['emotion'].value_counts()
    valid_emotions = counts[counts >= 2].index

    if len(valid_emotions) < len(counts):
        dropped = set(counts.index) - set(valid_emotions)
        print(f"⚠️ Warning: Dropping emotions with < 2 samples: {dropped}")
        df = df[df['emotion'].isin(valid_emotions)].copy()

    print(f"Loaded {len(df)} samples across {len(valid_emotions)} emotion classes.")

    # --- Preprocessing ---
    # Features (X): Landmarks (columns 3 onwards)
    # Target (y): Emotion (column 0)
    X = df.iloc[:, 3:].values
    y_raw = df['emotion'].values

    # Encode Labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print(f"Training on emotions: {list(le.classes_)}")

    # --- Split Data (80/20 Stratified) ---
    # This ensures the full emotion range is represented in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )

    print(f"Split Summary: Train={len(X_train)}, Test={len(X_test)}")

    # Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Advanced Model Selection ---
    print("\n--- Starting Model Search ---")
    print("Comparing Random Forest, SVM, and Neural Networks (MLP)...")

    models_config = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        },
        'SVM': {
            'model': SVC(random_state=42, probability=True),
            'params': {
                'C': [1, 10, 100],
                'kernel': ['rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        },
        'NeuralNet_MLP': {
            # max_iter is essentially "epochs"
            'model': MLPClassifier(random_state=42, max_iter=1000, early_stopping=True),
            'params': {
                'hidden_layer_sizes': [(128,), (128, 64), (256, 128)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001], # L2 regularization
                'learning_rate_init': [0.001, 0.01]
            }
        }
    }

    best_score = 0
    best_model = None
    best_model_name = ""

    for name, config in models_config.items():
        print(f"\nTraining {name}...")
        grid = GridSearchCV(config['model'], config['params'], cv=3, n_jobs=-1, verbose=1)
        grid.fit(X_train_scaled, y_train)
        
        print(f"  Best Params: {grid.best_params_}")
        print(f"  Best CV Score: {grid.best_score_:.4f}")
        
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
            best_model_name = name

    print(f"\n🏆 Best Model Selected: {best_model_name} (CV Score: {best_score:.4f})")

    # --- Final Evaluation ---
    print(f"\n--- Final Evaluation on Test Set ({best_model_name}) ---")
    y_pred = best_model.predict(X_test_scaled)

    print(f"Total Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    # --- Save Artifacts ---
    print(f"\n--- Saving Artifacts to {MODELS_DIR} ---")
    joblib.dump(best_model, MODEL_SAVE_PATH)
    joblib.dump(le, ENCODER_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print("Training pipeline complete.")

if __name__ == "__main__":
    train()