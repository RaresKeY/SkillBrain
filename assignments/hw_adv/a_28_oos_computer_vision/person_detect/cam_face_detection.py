import cv2
import numpy as np
from ultralytics import YOLO
import math
import os
import torch
import time

# ---------------- CONFIGURATION ---------------- #
MODEL_PATH = 'models/yolov8x_person_face.pt'
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# UI Settings
WINDOW_NAME = 'Person & Face Detection'
UI_WIDTH = 1280
UI_HEIGHT = 720
VIDEO_AREA_WIDTH = 900
VIDEO_AREA_HEIGHT = 600
VIDEO_POS_X = 20
VIDEO_POS_Y = 100

# Colors for classes (BGR)
COLORS = {
    'person': (255, 100, 0), # Blue-ish
    'face': (0, 255, 100)    # Green-ish
}
# ----------------------------------------------- #

def process_detection():
    # Change working directory to script location to handle relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Check for GPU
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    model.to(device)
    class_names = model.names

    # Initialize Video Capture (Webcam by default)
    # Use 0 for webcam, or provide path to video file
    video_source = 0 
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    cv2.namedWindow(WINDOW_NAME)
    
    # Create fixed UI background
    ui_background = np.zeros((UI_HEIGHT, UI_WIDTH, 3), dtype=np.uint8)
    ui_background[:] = (30, 30, 30) # Dark gray background
    
    # Header
    cv2.rectangle(ui_background, (0, 0), (UI_WIDTH, 80), (50, 50, 50), -1)
    cv2.putText(ui_background, "PERSON & FACE AI DETECTION", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    # Sidebar Area
    sidebar_x = VIDEO_POS_X + VIDEO_AREA_WIDTH + 20
    cv2.rectangle(ui_background, (sidebar_x, 100), (UI_WIDTH - 20, UI_HEIGHT - 20), (40, 40, 40), -1)

    print("Starting detection. Press 'q' to exit.")

    fps_list = []
    
    while True:
        start_time = time.time()
        success, frame = cap.read()
        if not success:
            break

        # Copy the static background
        display_img = ui_background.copy()

        # Resize frame to fit into VIDEO_AREA
        h, w = frame.shape[:2]
        scale = min(VIDEO_AREA_WIDTH / w, VIDEO_AREA_HEIGHT / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_frame = cv2.resize(frame, (new_w, new_h))
        
        # Inference
        results = model(resized_frame, stream=True, verbose=False, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)

        counts = {'person': 0, 'face': 0}
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls_idx = int(box.cls[0])
                label = class_names[cls_idx]
                
                counts[label] = counts.get(label, 0) + 1
                color = COLORS.get(label, (255, 255, 255))

                # Draw BBox
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw Label
                display_label = f'{label} {conf}'
                t_size = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(resized_frame, (x1, y1 - 20), (x1 + t_size[0], y1), color, -1)
                cv2.putText(resized_frame, display_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Composite frame onto UI
        y_offset = VIDEO_POS_Y + (VIDEO_AREA_HEIGHT - new_h) // 2
        x_offset = VIDEO_POS_X + (VIDEO_AREA_WIDTH - new_w) // 2
        display_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame
        
        # UI Borders
        cv2.rectangle(display_img, (VIDEO_POS_X-2, VIDEO_POS_Y-2), (VIDEO_POS_X + VIDEO_AREA_WIDTH + 2, VIDEO_POS_Y + VIDEO_AREA_HEIGHT + 2), (100, 100, 100), 2)

        # Statistics in Sidebar
        sb_x_text = sidebar_x + 20
        cv2.putText(display_img, "LIVE STATISTICS", (sb_x_text, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        
        row_y = 180
        for name, count in counts.items():
            color = COLORS.get(name, (255, 255, 255))
            cv2.putText(display_img, f"{name.upper()}: {count}", (sb_x_text, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            row_y += 40

        # FPS Calculation
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        fps_list.append(fps)
        if len(fps_list) > 30: fps_list.pop(0)
        avg_fps = sum(fps_list) / len(fps_list)
        
        cv2.putText(display_img, f"Avg FPS: {avg_fps:.1f}", (sb_x_text, UI_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(WINDOW_NAME, display_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_detection()
