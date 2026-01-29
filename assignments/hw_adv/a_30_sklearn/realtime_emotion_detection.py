
import cv2
import mediapipe as mp
import joblib
import numpy as np
import time
from pathlib import Path
from normalization import normalize_landmarks

# --- Configuration ---
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "face_landmarker.task"

CONFIDENCE_THRESHOLD = 0.6

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
ACTIVE_PRESET = "default"

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
    running_mode=VisionRunningMode.IMAGE, # Using IMAGE mode for simplicity in sync loop
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
)

def draw_landmarks(image, landmarks):
    h, w, c = image.shape
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting video stream... Press 'q' to exit.")

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to MediaPipe Image
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

            # Detect
            detection_result = landmarker.detect(mp_image)

            if detection_result.face_landmarks:
                landmarks = detection_result.face_landmarks[0]
                
                # Draw landmarks
                draw_landmarks(frame, landmarks)

                # Normalize & Predict
                h, w, c = frame.shape
                flat_landmarks = normalize_landmarks(landmarks, w, h)
                X = flat_landmarks.reshape(1, -1)
                X_scaled = scaler.transform(X)

                # Prediction Logic
                display_texts = []
                try:
                    probas = clf.predict_proba(X_scaled)[0]
                    # Get all indices that cross threshold
                    top_indices = np.where(probas >= CONFIDENCE_THRESHOLD)[0]
                    
                    if len(top_indices) > 0:
                        # Sort by confidence descending
                        sorted_indices = top_indices[np.argsort(probas[top_indices])][::-1]
                        for idx in sorted_indices:
                            emotion = le.inverse_transform([idx])[0]
                            conf = probas[idx]
                            display_texts.append(f"{emotion}: {conf:.2f}")
                    else:
                        # Fallback to absolute best if none cross threshold
                        best_idx = np.argmax(probas)
                        emotion = le.inverse_transform([best_idx])[0]
                        conf = probas[best_idx]
                        display_texts.append(f"({emotion}: {conf:.2f})")
                except:
                    # Fallback for models that don't support predict_proba
                    prediction_index = clf.predict(X_scaled)[0]
                    emotion = le.inverse_transform([prediction_index])[0]
                    display_texts.append(emotion)

                # Display Text Overlay
                box_height = 30 + (len(display_texts) * 35)
                cv2.rectangle(frame, (0,0), (300, box_height), (0,0,0), -1)
                
                for i, text in enumerate(display_texts):
                    y_pos = 35 + (i * 35)
                    cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # --- Normalized Landmarks Preview (Bottom Right) ---
                p_size = 150
                p_canvas = np.zeros((p_size, p_size, 3), dtype=np.uint8)
                pts = flat_landmarks.reshape(-1, 3)
                p_scale = p_size * 2
                for pt in pts:
                    px = int(pt[0] * p_scale + p_size / 2)
                    py = int(-pt[1] * p_scale + p_size / 2) # Flip Y back for display
                    if 0 <= px < p_size and 0 <= py < p_size:
                        cv2.circle(p_canvas, (px, py), 1, (0, 255, 255), -1)
                
                # Overlay on frame
                frame[h-p_size:h, w-p_size:w] = p_canvas
                cv2.rectangle(frame, (w-p_size, h-p_size), (w-1, h-1), (255, 255, 255), 1)

            cv2.imshow('Emotion Detector', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
