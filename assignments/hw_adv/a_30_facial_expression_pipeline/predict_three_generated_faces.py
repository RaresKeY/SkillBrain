"""Run emotion prediction on 3 curated generated-face images.

Reuses the a_30 pipeline approach:
- MediaPipe FaceLandmarker extraction
- normalize_landmarks from normalization.py
- trained sklearn model + scaler + label encoder from models/
"""

from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

from normalization import normalize_landmarks


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset" / "generated_faces"
EXPORT_DIR = BASE_DIR / "export" / "generated_face_predictions"


# Keep same preset structure used in a_30/test_model.py
PRESETS = {
    "combined_best": {
        "model": "emotion_model_combined.pkl",
        "encoder": "label_encoder_combined.pkl",
        "scaler": "scaler_combined.pkl",
    }
}

ACTIVE_PRESET = "combined_best"

# Curated filenames: neutral/common titles, no odd-tag edge cases.
SAMPLE_FILES = [
    "male_middle_african_neutral_average.png",
    "female_elderly_asian_sad_average.png",
    "male_young_indian_happy_average.png",
]


def load_models() -> tuple[object, object, object]:
    preset = PRESETS[ACTIVE_PRESET]
    clf = joblib.load(MODELS_DIR / preset["model"])
    le = joblib.load(MODELS_DIR / preset["encoder"])
    scaler = joblib.load(MODELS_DIR / preset["scaler"])
    return clf, le, scaler


def build_landmarker() -> mp.tasks.vision.FaceLandmarker:
    base_options = mp.tasks.BaseOptions(model_asset_path=str(MODELS_DIR / "face_landmarker.task"))
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def annotate_and_save(
    src_path: Path,
    pred_label: str,
    conf: float | None,
    face_bbox: tuple[int, int, int, int] | None,
    out_path: Path,
) -> None:
    def draw_rounded_rect(
        canvas: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        radius: int,
        color: tuple[int, int, int],
        thickness: int = -1,
    ) -> None:
        radius = max(2, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
        if thickness < 0:
            cv2.rectangle(canvas, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(canvas, (x1, y1 + radius), (x2, y2 - radius), color, -1)
            cv2.circle(canvas, (x1 + radius, y1 + radius), radius, color, -1)
            cv2.circle(canvas, (x2 - radius, y1 + radius), radius, color, -1)
            cv2.circle(canvas, (x1 + radius, y2 - radius), radius, color, -1)
            cv2.circle(canvas, (x2 - radius, y2 - radius), radius, color, -1)
        else:
            cv2.rectangle(canvas, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
            cv2.rectangle(canvas, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
            cv2.ellipse(canvas, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
            cv2.ellipse(canvas, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
            cv2.ellipse(canvas, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
            cv2.ellipse(canvas, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

    img = cv2.imread(str(src_path))
    if img is None:
        return
    _, w = img.shape[:2]

    # Modern rounded translucent overlay card (no full-width bar).
    title = "A30 Emotion Prediction"
    pred_line = f"pred: {pred_label}" + (f" ({conf:.0%})" if conf is not None else "")

    title_font = cv2.FONT_HERSHEY_SIMPLEX
    pred_font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 0.95
    pred_scale = 1.2
    title_thick = 2
    pred_thick = 3

    title_size, _ = cv2.getTextSize(title, title_font, title_scale, title_thick)
    pred_size, _ = cv2.getTextSize(pred_line, pred_font, pred_scale, pred_thick)

    pad_x = 18
    pad_y = 14
    line_gap = 14
    plate_w = min(w - 20, max(title_size[0], pred_size[0]) + pad_x * 2)
    plate_h = title_size[1] + pred_size[1] + pad_y * 2 + line_gap + 12
    plate_x1, plate_y1 = 14, 14
    plate_x2, plate_y2 = plate_x1 + plate_w, plate_y1 + plate_h

    # Brown translucent plate with clean yellow rounded border.
    overlay = img.copy()
    brown_fill = (38, 34, 28)   # BGR
    yellow_border = (40, 220, 245)  # BGR
    draw_rounded_rect(overlay, plate_x1, plate_y1, plate_x2, plate_y2, 18, brown_fill, -1)
    img = cv2.addWeighted(overlay, 0.84, img, 0.16, 0)
    # Border via outer+inner fill to avoid corner artifact lines.
    draw_rounded_rect(img, plate_x1, plate_y1, plate_x2, plate_y2, 18, yellow_border, -1)
    draw_rounded_rect(img, plate_x1 + 2, plate_y1 + 2, plate_x2 - 2, plate_y2 - 2, 16, brown_fill, -1)

    title_x = plate_x1 + pad_x
    title_y = plate_y1 + pad_y + title_size[1]
    pred_x = plate_x1 + pad_x
    pred_y = title_y + line_gap + pred_size[1]

    cv2.putText(
        img,
        title,
        (title_x, title_y),
        title_font,
        title_scale,
        (220, 235, 255),
        title_thick,
        cv2.LINE_AA,
    )
    # subtle shadow for pred line
    cv2.putText(
        img,
        pred_line,
        (pred_x + 2, pred_y + 2),
        pred_font,
        pred_scale,
        (8, 14, 24),
        pred_thick + 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        pred_line,
        (pred_x, pred_y),
        pred_font,
        pred_scale,
        (80, 225, 250),
        pred_thick,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(out_path), img)


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    clf, le, scaler = load_models()

    print(f"Using preset: {ACTIVE_PRESET}")
    print("Running emotion inference on 3 curated generated-face samples:\n")

    with build_landmarker() as landmarker:
        for idx, name in enumerate(SAMPLE_FILES, start=1):
            img_path = DATASET_DIR / name
            if not img_path.exists():
                print(f"[WARN] missing image: {img_path}")
                continue

            bgr = cv2.imread(str(img_path))
            if bgr is None:
                print(f"[WARN] cannot read image: {img_path}")
                continue

            h, w = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if not result.face_landmarks:
                print(f"{name} -> No face detected")
                out_path = EXPORT_DIR / f"a_30_pred_{idx:02d}_{img_path.stem}.png"
                annotate_and_save(img_path, "no_face", None, None, out_path)
                continue

            landmarks = result.face_landmarks[0]
            arr = np.array([(pt.x * w, pt.y * h) for pt in landmarks], dtype=np.float32)
            x1, y1 = np.maximum(arr.min(axis=0).astype(int) - 8, 0)
            x2, y2 = np.minimum(arr.max(axis=0).astype(int) + 8, [w - 1, h - 1])

            features = normalize_landmarks(landmarks, w, h).reshape(1, -1)
            X_scaled = scaler.transform(features)
            pred_idx = int(clf.predict(X_scaled)[0])
            pred_label = str(le.inverse_transform([pred_idx])[0])

            conf = None
            if hasattr(clf, "predict_proba"):
                conf = float(np.max(clf.predict_proba(X_scaled)[0]))
                print(f"{name} -> {pred_label} ({conf:.2f})")
            else:
                print(f"{name} -> {pred_label}")

            out_path = EXPORT_DIR / f"a_30_pred_{idx:02d}_{img_path.stem}.png"
            annotate_and_save(img_path, pred_label, conf, (x1, y1, x2, y2), out_path)
            print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
