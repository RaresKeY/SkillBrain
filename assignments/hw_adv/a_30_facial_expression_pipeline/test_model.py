
import cv2
import mediapipe as mp
import joblib
import numpy as np
from pathlib import Path
from normalization import normalize_landmarks
from helpers.folder_tools import file_generator, ft

# --- Configuration ---
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
TEST_IMAGES_DIR = BASE_DIR / "dataset" / "random_expressions"
MODEL_PATH = MODELS_DIR / "face_landmarker.task"

# --- Model Presets ---
PRESETS = {
    "default": {
        "model": "emotion_model.pkl",
        "encoder": "label_encoder.pkl",
        "scaler": "scaler.pkl"
    },
    "limited": {
        "model": "emotion_model_limited.pkl",
        "encoder": "label_encoder_limited.pkl",
        "scaler": "scaler_limited.pkl"
    },
    "limited_rf": {
        "model": "emotion_model_limited_rf.pkl",
        "encoder": "label_encoder_limited.pkl",
        "scaler": "scaler_limited.pkl"
    },
    "limited_svm": {
        "model": "emotion_model_limited_svm.pkl",
        "encoder": "label_encoder_limited.pkl",
        "scaler": "scaler_limited.pkl"
    },
    "limited_mlp": {
        "model": "emotion_model_limited_mlp.pkl",
        "encoder": "label_encoder_limited.pkl",
        "scaler": "scaler_limited.pkl"
    },
    "filtered": {
        "model": "emotion_model_filtered.pkl",
        "encoder": "label_encoder_filtered.pkl",
        "scaler": "scaler_filtered.pkl"
    },
    "filtered_rf": {
        "model": "emotion_model_filtered_rf.pkl",
        "encoder": "label_encoder_filtered.pkl",
        "scaler": "scaler_filtered.pkl"
    },
    "filtered_svm": {
        "model": "emotion_model_filtered_svm.pkl",
        "encoder": "label_encoder_filtered.pkl",
        "scaler": "scaler_filtered.pkl"
    },
    "filtered_mlp": {
        "model": "emotion_model_filtered_mlp.pkl",
        "encoder": "label_encoder_filtered.pkl",
        "scaler": "scaler_filtered.pkl"
    },
    "combined_best": {
        "model": "emotion_model_combined.pkl",
        "encoder": "label_encoder_combined.pkl",
        "scaler": "scaler_combined.pkl"
    },
    "combined_rf": {
        "model": "emotion_model_combined_rf.pkl",
        "encoder": "label_encoder_combined.pkl",
        "scaler": "scaler_combined.pkl"
    },
    "combined_svm": {
        "model": "emotion_model_combined_svm.pkl",
        "encoder": "label_encoder_combined.pkl",
        "scaler": "scaler_combined.pkl"
    },
    "combined_mlp": {
        "model": "emotion_model_combined_mlp.pkl",
        "encoder": "label_encoder_combined.pkl",
        "scaler": "scaler_combined.pkl"
    }
}

# Change this to switch models
ACTIVE_PRESET = "combined_best"
# ACTIVE_PRESET = "default"

print(f"--- Using Preset: {ACTIVE_PRESET.upper()} ---")
preset = PRESETS.get(ACTIVE_PRESET, PRESETS["default"])

# --- Load Models ---
try:
    clf = joblib.load(MODELS_DIR / preset["model"])
    le = joblib.load(MODELS_DIR / preset["encoder"])
    scaler = joblib.load(MODELS_DIR / preset["scaler"])
    print(f"Models loaded successfully ({preset['model']}).")
except FileNotFoundError as e:
    print(f"Error loading models: {e}")
    print(f"Make sure you have run the appropriate training script and have the files in {MODELS_DIR}")
    exit(1)

# --- MediaPipe Setup ---

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False, 
)

def predict_emotion(image_path, landmarker):
    cv_img = cv2.imread(str(image_path))
    if cv_img is None:
        print(f"Failed to load image: {image_path}")
        return

    # Convert to MediaPipe Image
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

    # Detect
    detection_result = landmarker.detect(mp_image)

    if detection_result.face_landmarks:
        # Assuming single face
        landmarks = detection_result.face_landmarks[0]
        h, w, c = cv_img.shape

        # Normalize
        flat_landmarks = normalize_landmarks(landmarks, w, h)
        
        # Scale
        # Reshape to 1 sample, n features
        X = flat_landmarks.reshape(1, -1)
        X_scaled = scaler.transform(X)

        # Predict
        prediction_index = clf.predict(X_scaled)[0]
        emotion = le.inverse_transform([prediction_index])[0]
        
        try:
            probabilities = clf.predict_proba(X_scaled)[0]
            confidence = np.max(probabilities)
            print(f"File: {image_path.name:<20} -> Emotion: {emotion:<15} (Confidence: {confidence:.2f})")
        except AttributeError:
             print(f"File: {image_path.name:<20} -> Emotion: {emotion:<15}")

    else:
        print(f"File: {image_path.name:<20} -> No face detected.")

def main():
    if not TEST_IMAGES_DIR.exists():
        print(f"Test directory not found: {TEST_IMAGES_DIR}")
        return

    print(f"\nTesting images from: {TEST_IMAGES_DIR}\n")
    
    with FaceLandmarker.create_from_options(options) as landmarker:
        # Get all images using folder_tools
        image_files = list(file_generator(TEST_IMAGES_DIR, ft.IMG))
        
        if not image_files:
            print("No images found in test directory.")
            return

        for img_path in image_files:
            predict_emotion(img_path, landmarker)

if __name__ == "__main__":
    main()
