import cv2
import numpy as np
from ultralytics import YOLO
import math
import os
import json
import torch
import copy

# ---------------- CONFIGURATION ---------------- #
MODEL_PATHS = [
    'models/fire_best.pt',
    'models/firedetect-11x.pt',
]
VIDEO_SOURCES = [
    # Add local dataset paths via .local_paths.json -> fire_video_sources
    "dataset/smoke_detection_false_positives_test/fp_clouds_haze/false_positive",
    "dataset/smoke_detection_false_positives_test/fp_dust_particles/false_positive",
]
CONF_THRESHOLD = 0.4
OUTPUT_JSON = "detection_results_gpu.json"
OUTPUT_JSON_SUMMARY = "detection_summary_gpu.json"
FRAME_SKIP = 20
TARGET_MP = 0.5 # Target Megapixels for AI inference (e.g., 0.5 MP ~= 900x580)

# UI Settings
UI_WIDTH = 1280
UI_HEIGHT = 720
VIDEO_AREA_WIDTH = 900
VIDEO_AREA_HEIGHT = 580
VIDEO_POS_X = 20
VIDEO_POS_Y = 100

# Speed Control Settings
DEFAULT_SPEED_VAL = 300 # Represents 1x (approx 30fps base)
SPEED_SLIDER_MAX = 100 # Map 0-100 to 0.25x - 1000x non-linearly
WINDOW_NAME = 'Smoke AI Detection Interface'
# ----------------------------------------------- #

def nothing(x):
    pass

def process_video(video_path, model, model_name, frame_skip=1, video_idx=0, total_videos=0, model_idx=0, total_models=0):
    filename = os.path.basename(video_path)
    folder_name = os.path.basename(os.path.dirname(video_path))
    print(f"Processing: {filename} with Model: {model_name} (Video {video_idx + 1}/{total_videos}, Model {model_idx + 1}/{total_models})")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0: fps = 30 # Fallback

    # Base delay for 1x speed
    base_delay = int(1000 / fps)

    classNames = model.names
    frame_count = 0
    frames_with_detections = 0

    # Initialize cumulative counts for all classes in the model
    cumulative_counts = {name: 0 for name in classNames.values()}

    # Data collection for this video
    video_data = {
        "filename": filename,
        "path": video_path,
        "model": model_name,
        "has_detection": False,
        "detections": []
    }

    # Create fixed UI background
    ui_background = np.zeros((UI_HEIGHT, UI_WIDTH, 3), dtype=np.uint8)
    # Dark gray background
    ui_background[:] = (30, 30, 30) 

    # Draw static UI elements (Headers/Boxes)
    # Header Area
    cv2.rectangle(ui_background, (0, 0), (UI_WIDTH, 80), (50, 50, 50), -1)
    cv2.putText(ui_background, "SMOKE & FIRE AI DETECTION", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # Right Sidebar Area (Background)
    sidebar_x = VIDEO_POS_X + VIDEO_AREA_WIDTH + 20
    cv2.rectangle(ui_background, (sidebar_x, 100), (UI_WIDTH - 20, UI_HEIGHT - 20), (40, 40, 40), -1)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        
        # Skip frames logic
        if frame_count % frame_skip != 0:
            continue
            
        # Copy the static background for the current frame
        display_img = ui_background.copy()

        # --- 1. PROCESS FRAME (Resize & Inference) ---
        # Target MP resize for Inference
        h, w = frame.shape[:2]
        current_pixels = w * h
        target_pixels = TARGET_MP * 1_000_000
        scale_factor = math.sqrt(target_pixels / current_pixels)

        inf_w = int(w * scale_factor)
        inf_h = int(h * scale_factor)
        inference_frame = cv2.resize(frame, (inf_w, inf_h))

        # Inference
        results = model(inference_frame, stream=True, verbose=False, conf=CONF_THRESHOLD)

        detected_in_frame = False
        frame_detections = []

        # Draw detections on inference_frame
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                current_class = classNames[cls]

                # Count every detection
                cumulative_counts[current_class] = cumulative_counts.get(current_class, 0) + 1

                if current_class.lower() in ['smoke', 'fire', 'flame', 'fire-smoke', 'factory-smoke']:
                    color = (0, 0, 255) # Red
                    detected_in_frame = True
                    video_data["has_detection"] = True
                    
                    frame_detections.append({
                        "class": current_class,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2] # Coordinates relative to inference_frame
                    })
                else:
                    color = (0, 255, 0) # Green

                cv2.rectangle(inference_frame, (x1, y1), (x2, y2), color, 2)
                # Label
                label = f'{current_class} {conf}'
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                c2 = x1 + t_size[0] + 2, y1 + t_size[1] + 5
                cv2.rectangle(inference_frame, (x1, y1), c2, color, -1)
                cv2.putText(inference_frame, label, (x1 + 1, y1 + t_size[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if detected_in_frame:
            frames_with_detections += 1
            video_data["detections"].append({
                "frame": frame_count,
                "timestamp_sec": round(frame_count / fps, 2),
                "objects": frame_detections
            })
            if frame_count % 30 == 0:
                 print(f"ALERT: Detection in frame {frame_count}")

        # --- 2. COMPOSITE FRAME ONTO UI ---
        # Resize inference frame to fit VIDEO_AREA for display
        # We use inference_frame as source now
        h_inf, w_inf = inference_frame.shape[:2]
        scale_disp = min(VIDEO_AREA_WIDTH / w_inf, VIDEO_AREA_HEIGHT / h_inf)
        new_w = int(w_inf * scale_disp)
        new_h = int(h_inf * scale_disp)
        display_video_frame = cv2.resize(inference_frame, (new_w, new_h))

        # Calculate centering offsets
        y_offset = VIDEO_POS_Y + (VIDEO_AREA_HEIGHT - new_h) // 2
        x_offset = VIDEO_POS_X + (VIDEO_AREA_WIDTH - new_w) // 2

        # Place video
        display_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = display_video_frame
        
        # Draw Border around video area
        cv2.rectangle(display_img, (VIDEO_POS_X-2, VIDEO_POS_Y-2), (VIDEO_POS_X + VIDEO_AREA_WIDTH + 2, VIDEO_POS_Y + VIDEO_AREA_HEIGHT + 2), (100, 100, 100), 2)


        # --- 3. DRAW DYNAMIC UI TEXT ---

        # Header Info
        header_info = f"Model: {model_name} | Folder: {folder_name}"
        t_size_h = cv2.getTextSize(header_info, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.putText(display_img, header_info, (UI_WIDTH - t_size_h[0] - 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        file_info = f"File: {filename} [Video {video_idx + 1}/{total_videos} | Model {model_idx + 1}/{total_models}]"
        t_size_f = cv2.getTextSize(file_info, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.putText(display_img, file_info, (UI_WIDTH - t_size_f[0] - 20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Sidebar Info
        sb_x_text = sidebar_x + 20
        cv2.putText(display_img, "STATUS", (sb_x_text, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

        if detected_in_frame:
            cv2.rectangle(display_img, (sb_x_text, 160), (sb_x_text + 200, 210), (0, 0, 255), -1)
            cv2.putText(display_img, "DETECTED", (sb_x_text + 10, 195), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        else:
            cv2.rectangle(display_img, (sb_x_text, 160), (sb_x_text + 220, 210), (0, 100, 0), -1)
            cv2.putText(display_img, "MONITORING", (sb_x_text + 10, 195), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.putText(display_img, "STATISTICS", (sb_x_text, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        cv2.putText(display_img, f"Frame: {frame_count} / {total_frames}", (sb_x_text, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(display_img, f"Det. Frames: {frames_with_detections}", (sb_x_text, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.putText(display_img, "CUMULATIVE DETECTIONS", (sb_x_text, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        row_y = 410
        for name, count in cumulative_counts.items():
            cv2.putText(display_img, f"{name}: {count}", (sb_x_text, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            row_y += 30

        # Progress Bar (Bottom of Video Area)
        bar_x = VIDEO_POS_X
        bar_y = VIDEO_POS_Y + VIDEO_AREA_HEIGHT + 15
        bar_w = VIDEO_AREA_WIDTH
        bar_h = 10

        # Draw background bar
        cv2.rectangle(display_img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)

        if total_frames > 0:
            progress = frame_count / total_frames
            fill_w = int(bar_w * progress)
            cv2.rectangle(display_img, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 255, 0), -1)

        # Speed Control Logic
        # Get slider position (0 - 100)
        slider_val = cv2.getTrackbarPos('Speed', WINDOW_NAME)

        if slider_val == -1: # Trackbar doesn't exist yet (shouldn't happen if init in main, but safe check) 
             slider_val = DEFAULT_SPEED_VAL

        # Map slider to speed multiplier: 
        # 0-40: 0.25x to 1x
        # 40-100: 1x to 1000x (Exponential-ish)

        if slider_val <= 40:
            # Range 0.25 to 1.0
            # 0 -> 0.25, 40 -> 1.0
            speed_mult = 0.25 + (slider_val / 40.0) * 0.75
        else:
            # Range 1.0 to 1000.0
            # 41 -> 1.0, 100 -> 1000
            # Use power function for smoother control at low end
            # (val - 40) / 60 => 0 to 1
            norm_val = (slider_val - 40) / 60.0
            # 1 + (999 * (norm_val^3))
            speed_mult = 1.0 + 99.0 * (norm_val ** 3)

        current_delay = int(base_delay / speed_mult)
        if current_delay < 1:
            current_delay = 1

        # Display Speed
        cv2.putText(display_img, f"Speed: {speed_mult:.2f}x", (bar_x + bar_w - 150, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow(WINDOW_NAME, display_img)

        # Controls
        key = cv2.waitKey(current_delay) & 0xFF
        
        # Spacebar to Pause
        if key == 32: 
            cv2.putText(display_img, "PAUSED", (VIDEO_POS_X + VIDEO_AREA_WIDTH // 2 - 50, VIDEO_POS_Y + VIDEO_AREA_HEIGHT // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.imshow(WINDOW_NAME, display_img)
            while True:
                key2 = cv2.waitKey(100) & 0xFF
                if key2 == 32: # Space again to resume
                    break
                if key2 == ord('q'):
                    key = ord('q')
                    break
                if key2 == ord('n'):
                    key = ord('n')
                    break

        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return "QUIT"
        elif key == ord('n'):
            break

    cap.release()
    return video_data

def main():
    # Change working directory to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory changed to: {script_dir}")

    # Check for GPU
    if torch.cuda.is_available():
        device = 0 # Use index for GPU 0
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA is available. Using GPU: {device_name}")
    else:
        device = 'cpu'
        print("CUDA is NOT available. Using CPU.")

    all_results = []
    print("Starting Smoke Detection...")
    print(f"Saving detailed results to: {OUTPUT_JSON}")
    print(f"Saving summary results to: {OUTPUT_JSON_SUMMARY}")

    # --- UI INIT ---
    cv2.namedWindow(WINDOW_NAME)
    cv2.createTrackbar('Speed', WINDOW_NAME, DEFAULT_SPEED_VAL, SPEED_SLIDER_MAX, nothing)
    # ????????

    # 1. Gather all unique video files
    all_video_files = []
    for source in VIDEO_SOURCES:
        if os.path.isdir(source):
            video_exts = ('.mp4', '.avi', '.mkv', '.mov', '.webm')
            files = sorted([
                os.path.join(source, f) 
                for f in os.listdir(source) 
                if f.lower().endswith(video_exts)
            ])
            all_video_files.extend(files)
        else:
            all_video_files.append(source)
            
    # Remove duplicates if any
    all_video_files = sorted(list(set(all_video_files)))
    total_videos = len(all_video_files)
    
    # 2. Pre-load Models (to avoid reloading for every video)
    loaded_models = []
    for m_path in MODEL_PATHS:
        try:
            print(f"Loading model: {m_path}")
            m = YOLO(m_path)
            m.to(device)
            loaded_models.append({
                "model": m,
                "name": os.path.basename(m_path)
            })
        except Exception as e:
            print(f"Failed to load {m_path}: {e}")
            
    total_models = len(loaded_models)

    should_quit = False
    
    # 3. Outer Loop: Videos
    for v_idx, video_file in enumerate(all_video_files):
        print(f"\n--- Starting Video {v_idx + 1}/{total_videos}: {os.path.basename(video_file)} ---")
        
        # 4. Inner Loop: Models
        for m_idx, model_info in enumerate(loaded_models):
            result = process_video(
                video_file, 
                model_info['model'], 
                model_info['name'], 
                FRAME_SKIP, 
                v_idx, 
                total_videos,
                m_idx,
                total_models
            )
            
            if result == "QUIT":
                should_quit = True
                break
            if result:
                all_results.append(result)
        
        if should_quit:
            break

    cv2.destroyAllWindows() 
    
    # Save Full JSON report
    try:
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nProcessing complete. Detailed report saved to {OUTPUT_JSON}")
    except Exception as e:
        print(f"Error saving detailed JSON report: {e}")

    # Create and Save Summary JSON report
    summary_results = []
    for r in all_results:
        summary_item = copy.deepcopy(r)
        # Replace detailed detections list with a count of frames having detections
        if 'detections' in summary_item:
            summary_item['detection_count'] = len(summary_item['detections'])
            del summary_item['detections']
        summary_results.append(summary_item)

    try:
        with open(OUTPUT_JSON_SUMMARY, 'w') as f:
            json.dump(summary_results, f, indent=2)
        print(f"Summary report saved to {OUTPUT_JSON_SUMMARY}")
    except Exception as e:
        print(f"Error saving summary JSON report: {e}")

    print(f"Total entries (video x model): {len(all_results)}")
    print(f"Entries with detections: {sum(1 for r in all_results if r['has_detection'])}")

if __name__ == "__main__":
    main()
