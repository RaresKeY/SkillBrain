import cv2
import numpy as np
from ultralytics import YOLO
import math
import os
import torch
import time
import json
from pathlib import Path

# ---------------- CONFIGURATION ---------------- #
def _load_local_paths() -> dict:
    here = Path(__file__).resolve()
    for base in [here.parent, *here.parents]:
        cfg = base / ".local_paths.json"
        if cfg.exists():
            try:
                return json.loads(cfg.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
    return {}


_LOCAL_PATHS = _load_local_paths()

MODEL_PATH = 'models/yolov8x_person_face.pt'
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# AI Processing Resolution Limits (Megapixels)
AI_MIN_MP = 0.0
AI_MAX_MP = 0.35

# Display & Playback Settings
PREVIEW_MATCHES_AI_RES = True # If True, display shows exactly what AI sees (pixelated if low res)
LOOP_VIDEOS = True            # If False, moves to next file automatically when video ends

# Speed and Skip Settings
MIN_SPEED_PCT = 25
MAX_SPEED_PCT = 5_000
DEFAULT_SPEED_PCT = 200
FRAME_SKIP = 10  # Process every Nth frame (1 = all, 2 = half, etc.)

# Data Sources
DATA_SOURCES = _LOCAL_PATHS.get("person_detect_sources", ['people_dataset_img'])

# Supported Extensions
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', '.mov', '.mkv'}

# UI Settings
WINDOW_NAME = 'Media Person & Face Detection'
UI_WIDTH = 1280
UI_HEIGHT = 720
VIDEO_AREA_WIDTH = 900
VIDEO_AREA_HEIGHT = 580
VIDEO_POS_X = 20
VIDEO_POS_Y = 100

# Colors for classes (BGR)
COLORS = {
    'person': (255, 100, 0), # Blue-ish
    'face': (0, 255, 100)    # Green-ish
}
# ----------------------------------------------- #

def nothing(x):
    pass

def get_all_files(sources):
    all_files = []
    for source in sources:
        if not os.path.exists(source):
            print(f"Warning: Source path not found: {source}")
            continue

        for root, dirs, files in os.walk(source):
            for file in sorted(files):
                ext = os.path.splitext(file)[1].lower()
                if ext in VALID_EXTENSIONS:
                    all_files.append(os.path.join(root, file))

    return sorted(all_files)

def resize_to_mp(img, min_mp, max_mp):
    """
    Resizes image to be within the target megapixel range while maintaining aspect ratio.
    Returns: resized_img, scale_factor (relative to original img)
    """
    h, w = img.shape[:2]
    current_pixels = w * h
    current_mp = current_pixels / 1_000_000.0

    target_scale = 1.0

    if current_mp < min_mp:
        # Upscale needed (rare, but good for very small inputs)
        target_scale = math.sqrt(min_mp / current_mp)
    elif current_mp > max_mp:
        # Downscale needed (optimization)
        target_scale = math.sqrt(max_mp / current_mp)

    # If scale is very close to 1, skip resizing
    if 0.95 < target_scale < 1.05:
        return img, 1.0

    new_w = int(w * target_scale)
    new_h = int(h * target_scale)
    resized = cv2.resize(img, (new_w, new_h))
    return resized, target_scale

def process_media(model, class_names, file_path, file_index, total_files, ui_background):
    ext = os.path.splitext(file_path)[1].lower()
    is_video = ext in {'.mp4', '.avi', '.mov', '.mkv'}

    fps = 30.0 # Default
    if is_video:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"Error: Could not open {file_path}")
            return 'next'
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        if vid_fps > 0:
            fps = vid_fps
    else:
        # Load image
        frame = cv2.imread(file_path)
        if frame is None:
            print(f"Error: Could not read image {file_path}")
            return 'next'

    print(f"Processing ({file_index + 1}/{total_files}): {os.path.basename(file_path)}")

    frame_count = 0

    while True:
        loop_start = time.time()

        if is_video:
            success, frame = cap.read()
            if not success:
                if LOOP_VIDEOS:
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    # Next video
                    cap.release()
                    return 'next'

            frame_count += 1
            # Skip frames logic
            if frame_count % FRAME_SKIP != 0:
                continue
        else:
            if frame is None: break 

        # Copy the static background
        display_img = ui_background.copy()

        # Prepare Inference Frame (MP Constraints)
        inf_frame, inf_scale = resize_to_mp(frame, AI_MIN_MP, AI_MAX_MP)

        # Prepare Display Frame
        if PREVIEW_MATCHES_AI_RES:
            # Display exactly what the AI sees (scaled to fit UI)
            h, w = inf_frame.shape[:2]
            disp_scale = min(VIDEO_AREA_WIDTH / w, VIDEO_AREA_HEIGHT / h)
            disp_w = int(w * disp_scale)
            disp_h = int(h * disp_scale)
            # Use NN interpolation to highlight pixelation if low res
            display_frame = cv2.resize(inf_frame, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
            coord_scale = disp_scale
        else:
            # Display high-quality original (scaled to fit UI)
            h, w = frame.shape[:2]
            disp_scale = min(VIDEO_AREA_WIDTH / w, VIDEO_AREA_HEIGHT / h)
            disp_w = int(w * disp_scale)
            disp_h = int(h * disp_scale)
            display_frame = cv2.resize(frame, (disp_w, disp_h))
            coord_scale = disp_scale / inf_scale

        # Inference
        results = model(inf_frame, stream=True, verbose=False, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)

        counts = {'person': 0, 'face': 0}

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0] # Coords in inf_frame
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

                # Scale to Display Frame
                x1_d = int(x1 * coord_scale)
                y1_d = int(y1 * coord_scale)
                x2_d = int(x2 * coord_scale)
                y2_d = int(y2 * coord_scale)

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls_idx = int(box.cls[0])
                label = class_names[cls_idx]

                counts[label] = counts.get(label, 0) + 1
                color = COLORS.get(label, (255, 255, 255))

                # Draw BBox
                cv2.rectangle(display_frame, (x1_d, y1_d), (x2_d, y2_d), color, 2)

                # Draw Label
                display_label = f'{label} {conf}'
                t_size = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(display_frame, (x1_d, y1_d - 20), (x1_d + t_size[0], y1_d), color, -1)
                cv2.putText(display_frame, display_label, (x1_d, y1_d - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Composite frame onto UI
        y_offset = VIDEO_POS_Y + (VIDEO_AREA_HEIGHT - disp_h) // 2
        x_offset = VIDEO_POS_X + (VIDEO_AREA_WIDTH - disp_w) // 2
        display_img[y_offset:y_offset+disp_h, x_offset:x_offset+disp_w] = display_frame

        # UI Borders
        cv2.rectangle(display_img, (VIDEO_POS_X-2, VIDEO_POS_Y-2), (VIDEO_POS_X + VIDEO_AREA_WIDTH + 2, VIDEO_POS_Y + VIDEO_AREA_HEIGHT + 2), (100, 100, 100), 2)

        # File Info
        cv2.putText(display_img, f"File ({file_index + 1}/{total_files}): {os.path.basename(file_path)}", (VIDEO_POS_X, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Resolution Info
        inf_h, inf_w = inf_frame.shape[:2]
        inf_mp = (inf_w * inf_h) / 1_000_000.0

        orig_h, orig_w = frame.shape[:2]
        res_info = f"Orig: {orig_w}x{orig_h} | AI Input: {inf_w}x{inf_h} ({inf_mp:.2f} MP)"
        if PREVIEW_MATCHES_AI_RES:
            res_info += " [PREVIEW MATCHES AI]"

        cv2.putText(display_img, res_info, (VIDEO_POS_X, VIDEO_POS_Y + VIDEO_AREA_HEIGHT + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        # Sidebar Statistics
        sidebar_x = VIDEO_POS_X + VIDEO_AREA_WIDTH + 20
        sb_x_text = sidebar_x + 20
        cv2.putText(display_img, "LIVE STATISTICS", (sb_x_text, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

        row_y = 180
        for name, count in counts.items():
            color = COLORS.get(name, (255, 255, 255))
            cv2.putText(display_img, f"{name.upper()}: {count}", (sb_x_text, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            row_y += 40

        # Controls Info
        cv2.putText(display_img, "CONTROLS:", (sb_x_text, UI_HEIGHT - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        cv2.putText(display_img, "'n' - Next File", (sb_x_text, UI_HEIGHT - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display_img, "'p' - Prev File", (sb_x_text, UI_HEIGHT - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display_img, "'q' - Quit", (sb_x_text, UI_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Speed Calculation
        speed_pct = cv2.getTrackbarPos('Speed %', WINDOW_NAME)
        if speed_pct < MIN_SPEED_PCT: 
            speed_pct = MIN_SPEED_PCT
            cv2.setTrackbarPos('Speed %', WINDOW_NAME, MIN_SPEED_PCT)

        # Display Speed & Skip info
        speed_info = f"Speed: {speed_pct}%"
        if FRAME_SKIP > 1:
            speed_info += f" (Skip: {FRAME_SKIP})"
        cv2.putText(display_img, speed_info, (VIDEO_POS_X + VIDEO_AREA_WIDTH - 220, VIDEO_POS_Y + VIDEO_AREA_HEIGHT + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow(WINDOW_NAME, display_img)

        # Wait Key Logic
        if is_video:
            # Base delay for 1x speed
            base_delay = 1000.0 / fps
            # Adjusted delay
            delay = int(base_delay * (100.0 / speed_pct))
            if delay < 1: delay = 1
        else:
            delay = 100

        key = cv2.waitKey(delay) & 0xFF
        
        if key == ord('q'):
            if is_video: cap.release()
            return 'quit'
        elif key == ord('n'):
            if is_video: cap.release()
            return 'next'
        elif key == ord('p'):
            if is_video: cap.release()
            return 'prev'

    if is_video: cap.release()
    return 'next'

def main():
    # Change working directory to script location
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

    # Gather Files
    print("Scanning for media files...")
    media_files = get_all_files(DATA_SOURCES)

    if not media_files:
        print("No media files found in defined sources.")
        return

    print(f"Found {len(media_files)} files.")

    # UI Setup
    cv2.namedWindow(WINDOW_NAME)
    cv2.createTrackbar('Speed %', WINDOW_NAME, DEFAULT_SPEED_PCT, MAX_SPEED_PCT, nothing)

    ui_background = np.zeros((UI_HEIGHT, UI_WIDTH, 3), dtype=np.uint8)
    ui_background[:] = (30, 30, 30)

    # Static Header & Sidebar Background
    cv2.rectangle(ui_background, (0, 0), (UI_WIDTH, 80), (50, 50, 50), -1)
    cv2.putText(ui_background, "MEDIA INSPECTOR: PERSON & FACE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    sidebar_x = VIDEO_POS_X + VIDEO_AREA_WIDTH + 20
    cv2.rectangle(ui_background, (sidebar_x, 100), (UI_WIDTH - 20, UI_HEIGHT - 20), (40, 40, 40), -1)

    idx = 0
    total = len(media_files)

    while True:
        if idx < 0: idx = total - 1
        if idx >= total: idx = 0

        current_file = media_files[idx]
        action = process_media(model, class_names, current_file, idx, total, ui_background)

        if action == 'quit':
            break
        elif action == 'next':
            idx += 1
        elif action == 'prev':
            idx -= 1

    cv2.destroyAllWindows()
    print("Exiting...")

if __name__ == "__main__":
    main()
