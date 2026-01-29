from helpers.folder_tools import file_generator, ft
import mediapipe as mp
import cv2
import csv
import numpy as np
from pathlib import Path
from normalization import normalize_landmarks

PREDICTOR_PATH = "assignments/hw_adv/a_30_sklearn/models/shape_predictor_68_face_landmarks.dat"
OUTPUT_DATA = "assignments/hw_adv/a_30_sklearn/dataset/landmarks.csv"
MODEL_PATH = "assignments/hw_adv/a_30_sklearn/models/face_landmarker.task"

test_folder = Path("assignments/hw_adv/a_30_sklearn/dataset/generated_faces")
files = file_generator(test_folder, ft.IMG)

header = ["filename", "face_id"]
for face_id in range(68):
    header.extend([f"x_{face_id}", f"y_{face_id}"])

# --- Setup MediaPipe Tasks ---
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create the options with the model path
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True, # Set True if you want expression coefficients (smile score, etc)
)

with FaceLandmarker.create_from_options(options) as landmarker:
    # 1. Setup CSV Writer
    # The new model outputs 478 landmarks
    # MOVED "emotion" to the front
    header = ["emotion", "filename", "face_id"]
    for i in range(478):
        header.extend([f"x_{i}", f"y_{i}", f"z_{i}"])

    with open(OUTPUT_DATA, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # 2. Load Image (Standard OpenCV)
        # Replace this with your loop logic
        for image_path in files:
            cv_img = cv2.imread(image_path)

            if cv_img is not None:
                # 3. Convert to MediaPipe Image Object (CRITICAL NEW STEP)
                # OpenCV loads as BGR, MediaPipe Tasks expects RGB
                rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

                # 4. Detect
                detection_result = landmarker.detect(mp_image)

                # 5. Extract Data
                if detection_result.face_landmarks:
                    for face_id, landmarks in enumerate(detection_result.face_landmarks):
                        h, w, c = cv_img.shape
                        
                        # Use the helper function
                        flat_landmarks = normalize_landmarks(landmarks, w, h)

                        # Extract emotion from filename
                        # Format: sex_age_ethnicity_expression_attractiveness.png
                        filename = Path(image_path).name
                        try:
                            # Split by underscore, grab the second to last element (index -2) which corresponds to expression
                            # Format: sex_age_ethnicity_expression_attractiveness.png
                            # E.g. male_young_caucasian_happy_average.png -> happy (index -2)
                            # E.g. male_young_middle_eastern_neutral_average.png -> neutral (index -2)
                            parts = filename.replace(".png", "").split("_")
                            if len(parts) >= 4:
                                emotion = parts[-2]
                            else:
                                emotion = "unknown"
                        except Exception:
                            emotion = "unknown"

                        # emotion is now first
                        row = [emotion, image_path, face_id]
                        row.extend(flat_landmarks)
                        
                        writer.writerow(row)
                        print(f"Saved face {face_id} from {image_path} (Emotion: {emotion})")

                        # Store for visualization outside loop
                        last_landmarks = landmarks
                        last_image_shape = cv_img.shape
                else:
                    print(f"No face detected in {image_path}")