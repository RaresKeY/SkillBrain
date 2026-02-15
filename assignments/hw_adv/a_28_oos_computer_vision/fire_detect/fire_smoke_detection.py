import cv2
from ultralytics import YOLO
import math
import os
import json

# ---------------- CONFIGURATION ---------------- #
MODEL_PATH = 'assignments/hw_adv/a_28_computer_vision/fire_best.pt'
VIDEO_SOURCE = "assignments/hw_adv/a_28_computer_vision/dataset/smoke_detection_false_positives/fp_clouds_haze/false_positive"
CONF_THRESHOLD = 0.4
OUTPUT_JSON = "detection_results.json"
# ----------------------------------------------- #

def process_video(video_path, model):
    filename = os.path.basename(video_path)
    print(f"Processing: {filename}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 1000
    # if fps <= 0: 
    #     fps = 30
    delay = int(1000 / fps)

    classNames = model.names
    frame_count = 0

    # Data collection for this video
    video_data = {
        "filename": filename,
        "path": video_path,
        "has_detection": False,
        "detections": []
    }

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        # 1. Resize for AI if too large (manageable size)
        h, w = frame.shape[:2]
        max_dim = max(h, w)
        if max_dim > 1080:
            scale = 360 / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))

        # Run Inference
        results = model(frame, stream=True, verbose=False, conf=CONF_THRESHOLD)

        # 2. Add Border for "out of frame names" (labels)
        border_top = 40
        display_frame = cv2.copyMakeBorder(frame, border_top, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        detected_in_frame = False
        frame_detections = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box (relative to the resized 'frame')
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Confidence & Class
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                current_class = classNames[cls]

                # Alert logic
                if current_class.lower() in ['smoke', 'fire', 'flame']:
                    color = (0, 0, 255)
                    detected_in_frame = True
                    video_data["has_detection"] = True
                    
                    frame_detections.append({
                        "class": current_class,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2]
                    })
                    
                    if frame_count % 30 == 0:
                        print(f"ALERT: {current_class} detected! Confidence: {conf}")
                else:
                    color = (0, 255, 0)

                # Draw detections on 'display_frame' (shifted by border)
                # Shift y-coordinates
                dy1 = y1 + border_top
                dy2 = y2 + border_top
                dx1, dx2 = x1, x2

                cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2), color, 2)
                label = f'{current_class} {conf}'
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                c2 = dx1 + t_size[0], dy1 - t_size[1] - 5
                cv2.rectangle(display_frame, (dx1, dy1), c2, color, -1)
                cv2.putText(display_frame, label, (dx1, dy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], thickness=1)
        
        if detected_in_frame:
            video_data["detections"].append({
                "frame": frame_count,
                "timestamp_sec": round(frame_count / fps, 2),
                "objects": frame_detections
            })

        # Final resize for display if ANY dimension is still too large for screen (e.g. vertical video with border)
        # We reused the logic but applied it to display_frame
        d_h, d_w = display_frame.shape[:2]
        max_screen_w, max_screen_h = 1280, 800
        min_screen_w = 800
        
        scale = 1.0
        if d_w > max_screen_w or d_h > max_screen_h:
            scale = min(max_screen_w/d_w, max_screen_h/d_h)
        elif d_w < min_screen_w:
            scale = min_screen_w / d_w
            if (d_h * scale) > max_screen_h:
                scale = max_screen_h / d_h
        
        if scale != 1.0:
            new_w = int(d_w * scale)
            new_h = int(d_h * scale)
            display_frame = cv2.resize(display_frame, (new_w, new_h))

        cv2.imshow('Smoke AI Detection', display_frame)

        # Controls
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return "QUIT"
        elif key == ord('n'):
            break

    cap.release()
    return video_data

def main():
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Starting Smoke Detection...")
    print(f"Saving results to: {OUTPUT_JSON}")

    # Determine file list
    if os.path.isdir(VIDEO_SOURCE):
        video_exts = ('.mp4', '.avi', '.mkv', '.mov', '.webm')
        files = sorted([
            os.path.join(VIDEO_SOURCE, f) 
            for f in os.listdir(VIDEO_SOURCE) 
            if f.lower().endswith(video_exts)
        ])
    else:
        files = [VIDEO_SOURCE]

    all_results = []

    for video_file in files:
        result = process_video(video_file, model)
        if result == "QUIT":
            break
        if result:
            all_results.append(result)

    cv2.destroyAllWindows()
    
    # Save JSON report
    try:
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nProcessing complete. Report saved to {OUTPUT_JSON}")
        print(f"Total videos processed: {len(all_results)}")
        print(f"Videos with detections: {sum(1 for r in all_results if r['has_detection'])}")
    except Exception as e:
        print(f"Error saving JSON report: {e}")

if __name__ == "__main__":
    main()