import cv2
import numpy as np
from ultralytics import YOLO
import math
import os
import torch
from pathlib import Path
from typing import Union, List, Dict, Any

# ---------------- CONFIGURATION ---------------- #
CURRENT_DIR = Path(__file__).parent
DEFAULT_MODEL_PATH = CURRENT_DIR / "../../a_28_computer_vision/person_detect/models/yolov8x_person_face.pt"

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Colors for classes (BGR)
COLORS = {
    'person': (255, 100, 0), # Blue-ish
    'face': (0, 255, 100)    # Green-ish
}

_MODEL_CACHE = {}

def get_model(model_path: Union[str, Path]) -> YOLO:
    """Loads and caches the YOLO model."""
    path_str = str(Path(model_path).resolve())
    if path_str not in _MODEL_CACHE:
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Model not found at {path_str}")
        
        device = 0 if torch.cuda.is_available() else 'cpu'
        model = YOLO(path_str)
        model.to(device)
        _MODEL_CACHE[path_str] = model
    return _MODEL_CACHE[path_str]

def detect_face_person(
    image_input: Union[str, Path, np.ndarray], 
    model_path: Union[str, Path, None] = None,
    conf_threshold: float = CONF_THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    Detects persons and faces in a single image.

    Args:
        image_input: Path to image file or a numpy BGR image.
        model_path: Path to YOLO model. Defaults to internal project path.
        conf_threshold: Confidence threshold for detection.
        iou_threshold: IOU threshold for NMS.

    Returns:
        List of dictionaries containing:
            'label': class name (person/face)
            'confidence': detection score
            'box': [x1, y1, x2, y2]
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    model = get_model(model_path)
    class_names = model.names

    # Load image if path is provided
    if isinstance(image_input, (str, Path)):
        image = cv2.imread(str(image_input))
        if image is None:
            raise ValueError(f"Could not read image from {image_input}")
    else:
        image = image_input

    # Inference (Note: model handles resizing internally)
    results = model(image, verbose=False, conf=conf_threshold, iou=iou_threshold)

    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_idx = int(box.cls[0])
            label = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)

            detections.append({
                'label': label,
                'confidence': round(conf, 2),
                'box': [int(x1), int(y1), int(x2), int(y2)]
            })

    return detections

def detect_from_ndarray(
    image: np.ndarray, 
    model_path: Union[str, Path, None] = None,
    conf_threshold: float = CONF_THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    Detects persons and faces in a numpy BGR image.
    """
    return detect_face_person(image, model_path, conf_threshold, iou_threshold)

def draw_detections(image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """Draws bounding boxes and labels on the image."""
    output_img = image.copy()
    for det in detections:
        label = det['label']
        conf = det['confidence']
        x1, y1, x2, y2 = det['box']
        
        color = COLORS.get(label, (255, 255, 255))
        
        # Draw BBox
        cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
        
        # Draw Label
        display_label = f'{label} {conf}'
        t_size = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(output_img, (x1, y1 - 20), (x1 + t_size[0], y1), color, -1)
        cv2.putText(output_img, display_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    return output_img

if __name__ == "__main__":
    # Example usage:
    # results = detect_face_person("path/to/image.jpg")
    # print(results)
    pass