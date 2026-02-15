import sys
import os
import cv2
import time
import math
import numpy as np
import torch
import json
import copy

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QSlider, QProgressBar, 
                               QListWidget, QFrame, QSizePolicy, QPushButton, QHBoxLayout as QHBox)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QMutex, QWaitCondition
from PySide6.QtGui import QImage, QPixmap, QFont, QPainter

from ultralytics import YOLO

# ---------------- CONFIGURATION ---------------- #
MODEL_PATHS = [
    'models/fire_best.pt',
    'models/firedetect-11s.pt',
    'models/firedetect-11x.pt',
    'models/yolov8s-forest-fire-detection.pt',
]
VIDEO_SOURCES = [
    "/media/mintmainog/c21d735b-a894-4487-8dc4-b83f31f0a84c/fire_dataset/smoke_videos.1407/pos",
    # "dataset/smoke_detection_false_positives/fp_clouds_haze/false_positive",
    # "dataset/smoke_detection_false_positives/fp_dust_particles/false_positive",
]
CONF_THRESHOLD = 0.4
OUTPUT_JSON = "detection_results_pyside.json"
OUTPUT_JSON_SUMMARY = "detection_summary_pyside.json"
FRAME_SKIP = 1

# Change working directory to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
# ----------------------------------------------- #

class ImageLabel(QLabel):
    """Custom Label to handle aspect-ratio preserved image painting without forcing resize."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScaledContents(False) # Important: We handle scaling manually
        self._pixmap = None
        self.setStyleSheet("background-color: black; border: 2px solid #555;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Set a minimum size to prevent collapsing but allow shrinking
        self.setMinimumSize(200, 200)

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.update() # Schedule a repaint

    def paintEvent(self, event):
        if not self._pixmap or self._pixmap.isNull():
            super().paintEvent(event)
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # Calculate target rect maintaining aspect ratio
        widget_size = self.size()
        pixmap_size = self._pixmap.size()
        
        scaled_pixmap = self._pixmap.scaled(widget_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Center the image
        x = (widget_size.width() - scaled_pixmap.width()) // 2
        y = (widget_size.height() - scaled_pixmap.height()) // 2
        
        painter.drawPixmap(x, y, scaled_pixmap)

class VideoProcessor(QThread):
    change_pixmap_signal = Signal(QImage)
    update_stats_signal = Signal(dict)
    update_cumulative_signal = Signal(dict)
    update_progress_signal = Signal(int, int) # current, total
    video_finished_signal = Signal(dict) # Returns results for this video
    all_finished_signal = Signal()
    
    def __init__(self, model_paths, video_sources):
        super().__init__()
        self.model_paths = model_paths
        self.video_sources = video_sources
        self._run_flag = True
        self.speed_multiplier = 1.0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Navigation flags
        self.next_requested = False
        self.prev_requested = False
        
        print(f"Using Device: {self.device}")

    def run(self):
        # 1. Gather all unique video files
        all_video_files = []
        for source in self.video_sources:
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
        all_video_files = sorted(list(set(all_video_files)))
        
        # 2. Pre-load Models
        loaded_models = []
        for m_path in self.model_paths:
            try:
                m = YOLO(m_path)
                m.to(self.device)
                loaded_models.append({"model": m, "name": os.path.basename(m_path)})
            except Exception as e:
                print(f"Failed to load {m_path}: {e}")

        # 3. Create Flattened Task List (Video x Model)
        tasks = []
        for v_idx, video_file in enumerate(all_video_files):
            for m_idx, model_info in enumerate(loaded_models):
                tasks.append({
                    "video_file": video_file,
                    "model_info": model_info,
                    "v_idx": v_idx,
                    "m_idx": m_idx,
                    "total_videos": len(all_video_files),
                    "total_models": len(loaded_models)
                })
        
        total_tasks = len(tasks)
        current_task_idx = 0

        while current_task_idx < total_tasks and self._run_flag:
            task = tasks[current_task_idx]
            
            # Reset flags
            self.next_requested = False
            self.prev_requested = False
            
            result = self.process_single_video(
                task["video_file"], 
                task["model_info"]['model'], 
                task["model_info"]['name'],
                task["v_idx"], task["total_videos"], 
                task["m_idx"], task["total_models"]
            )
            
            if result and not self.prev_requested: # Only save if we finished naturally or skipped forward
                self.video_finished_signal.emit(result)

            # Navigation Logic
            if self.prev_requested:
                current_task_idx = max(0, current_task_idx - 1)
            elif self.next_requested:
                current_task_idx += 1
            else:
                # Normal completion
                current_task_idx += 1
        
        self.all_finished_signal.emit()

    def process_single_video(self, video_path, model, model_name, v_idx, total_videos, m_idx, total_models):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        filename = os.path.basename(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0: fps = 30
        
        base_delay_ms = int(1000 / fps)
        
        classNames = model.names
        frame_count = 0
        frames_with_detections = 0
        cumulative_counts = {name: 0 for name in classNames.values()}
        
        video_data = {
            "filename": filename,
            "path": video_path,
            "model": model_name,
            "has_detection": False,
            "detections": []
        }

        # Initial Stats Emit
        self.update_stats_signal.emit({
            "filename": filename,
            "model_name": model_name,
            "video_idx": v_idx + 1, "total_videos": total_videos,
            "model_idx": m_idx + 1, "total_models": total_models,
            "status": "MONITORING",
            "frame_count": 0, "total_frames": total_frames,
            "det_frames": 0
        })
        self.update_cumulative_signal.emit(cumulative_counts)

        while self._run_flag:
            # Check navigation
            if self.next_requested or self.prev_requested:
                break

            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue

            # Resize
            target_h = 600
            h, w = frame.shape[:2]
            scale = target_h / h
            new_w = int(w * scale)
            resized_frame = cv2.resize(frame, (new_w, target_h))

            # Inference
            results = model(resized_frame, verbose=False, conf=CONF_THRESHOLD)
            
            detected_in_frame = False
            frame_detections = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    current_class = classNames[cls]

                    # Count EVERY detection
                    cumulative_counts[current_class] = cumulative_counts.get(current_class, 0) + 1

                    if current_class.lower() in ['smoke', 'fire', 'flame', 'fire-smoke', 'factory-smoke']:
                        color = (0, 0, 255) # Red (BGR)
                        detected_in_frame = True
                        video_data["has_detection"] = True
                        
                        frame_detections.append({
                            "class": current_class,
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2]
                        })
                    else:
                        color = (0, 255, 0) # Green (BGR)

                    cv2.rectangle(resized_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Label inside box
                    label = f'{current_class} {conf}'
                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    c2 = x1 + t_size[0] + 2, y1 + t_size[1] + 5
                    cv2.rectangle(resized_frame, (x1, y1), c2, color, -1)
                    cv2.putText(resized_frame, label, (x1 + 1, y1 + t_size[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if detected_in_frame:
                frames_with_detections += 1
                video_data["detections"].append({
                    "frame": frame_count,
                    "timestamp_sec": round(frame_count / fps, 2),
                    "objects": frame_detections
                })

            # --- Update UI ---
            status = "DETECTED" if detected_in_frame else "MONITORING"
            self.update_stats_signal.emit({
                "filename": filename,
                "model_name": model_name,
                "video_idx": v_idx + 1, "total_videos": total_videos,
                "model_idx": m_idx + 1, "total_models": total_models,
                "status": status,
                "frame_count": frame_count, "total_frames": total_frames,
                "det_frames": frames_with_detections
            })
            
            if len(results) > 0: 
                self.update_cumulative_signal.emit(cumulative_counts)
                
            self.update_progress_signal.emit(frame_count, total_frames)

            # Convert to QImage
            rgb_image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.change_pixmap_signal.emit(convert_to_Qt_format)

            # Speed Control
            delay = int(base_delay_ms / self.speed_multiplier)
            if delay < 1: delay = 1
            self.msleep(delay)

        cap.release()
        # If interrupted by prev/next, return None so partial results aren't saved (optional choice)
        if self.next_requested or self.prev_requested:
            return None
        return video_data

    def stop(self):
        self._run_flag = False
        self.wait()

    def set_speed(self, val):
        if val <= 40:
            self.speed_multiplier = 0.25 + (val / 40.0) * 0.75
        else:
            norm_val = (val - 40) / 60.0
            self.speed_multiplier = 1.0 + 999.0 * (norm_val ** 3)

    def request_next(self):
        self.next_requested = True

    def request_prev(self):
        self.prev_requested = True


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smoke & Fire AI Detection - PySide6")
        self.resize(1400, 800)
        self.setStyleSheet("background-color: #2e2e2e; color: white;")

        # --- Data Holders ---
        self.all_results = []

        # --- Main Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 1. Video Display Area (Left)
        self.video_label = ImageLabel() 
        main_layout.addWidget(self.video_label, stretch=3)

        # 2. Controls & Stats Area (Right)
        right_panel = QFrame()
        right_panel.setFixedWidth(350)
        right_panel.setStyleSheet("background-color: #3e3e3e; border-left: 1px solid #555;")
        right_layout = QVBoxLayout(right_panel)
        main_layout.addWidget(right_panel, stretch=1)

        # --- Right Panel Elements ---
        
        # Header
        title = QLabel("DETECTION DASHBOARD")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(title)
        right_layout.addSpacing(10)

        # File Info
        self.lbl_model = QLabel("Model: -")
        self.lbl_file = QLabel("File: -")
        self.lbl_file.setWordWrap(True)
        self.lbl_progress = QLabel("Video: 0/0 | Model: 0/0")
        
        for lbl in [self.lbl_model, self.lbl_file, self.lbl_progress]:
            lbl.setFont(QFont("Arial", 11))
            right_layout.addWidget(lbl)
        
        right_layout.addSpacing(20)

        # Status Box
        self.lbl_status = QLabel("WAITING")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setFont(QFont("Arial", 14, QFont.Bold))
        self.lbl_status.setFixedHeight(50)
        self.lbl_status.setStyleSheet("background-color: gray; color: white; border-radius: 5px;")
        right_layout.addWidget(self.lbl_status)
        
        right_layout.addSpacing(20)

        # Stats
        self.lbl_frame_stats = QLabel("Frame: 0/0")
        self.lbl_det_stats = QLabel("Detected Frames: 0")
        for lbl in [self.lbl_frame_stats, self.lbl_det_stats]:
            lbl.setFont(QFont("Arial", 11))
            right_layout.addWidget(lbl)

        right_layout.addSpacing(20)

        # Cumulative List
        lbl_cum = QLabel("CUMULATIVE DETECTIONS:")
        lbl_cum.setFont(QFont("Arial", 12, QFont.Bold))
        right_layout.addWidget(lbl_cum)

        self.list_cumulative = QListWidget()
        self.list_cumulative.setStyleSheet("background-color: #222; border: 1px solid #555;")
        right_layout.addWidget(self.list_cumulative)

        right_layout.addSpacing(20)

        # Speed Control
        self.lbl_speed = QLabel("Speed: 1.00x")
        right_layout.addWidget(self.lbl_speed)
        
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(0, 100)
        self.slider_speed.setValue(40) # Default ~1x
        self.slider_speed.valueChanged.connect(self.update_speed)
        right_layout.addWidget(self.slider_speed)

        # Navigation Buttons
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("<< PREV")
        self.btn_next = QPushButton("NEXT >>")
        
        for btn in [self.btn_prev, self.btn_next]:
            btn.setFixedHeight(40)
            btn.setFont(QFont("Arial", 10, QFont.Bold))
            btn.setStyleSheet("background-color: #555; border-radius: 5px;")
            nav_layout.addWidget(btn)
            
        self.btn_prev.clicked.connect(self.prev_video)
        self.btn_next.clicked.connect(self.next_video)
        right_layout.addLayout(nav_layout)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #05B8CC; 
            }
        """)
        right_layout.addWidget(self.progress_bar)

        # --- Thread Setup ---
        self.thread = VideoProcessor(MODEL_PATHS, VIDEO_SOURCES)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_stats_signal.connect(self.update_stats)
        self.thread.update_cumulative_signal.connect(self.update_cumulative)
        self.thread.update_progress_signal.connect(self.update_progress)
        self.thread.video_finished_signal.connect(self.collect_result)
        self.thread.all_finished_signal.connect(self.save_results)
        
        # Start
        self.update_speed(self.slider_speed.value())
        self.thread.start()

    @Slot(QImage)
    def update_image(self, qt_img):
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    @Slot(dict)
    def update_stats(self, data):
        self.lbl_model.setText(f"Model: {data['model_name']}")
        self.lbl_file.setText(f"File: {data['filename']}")
        self.lbl_progress.setText(f"Video: {data['video_idx']}/{data['total_videos']} | Model: {data['model_idx']}/{data['total_models']}")
        
        status = data['status']
        self.lbl_status.setText(status)
        if status == "DETECTED":
            self.lbl_status.setStyleSheet("background-color: #cc0000; color: white; border-radius: 5px;")
        else:
            self.lbl_status.setStyleSheet("background-color: #009900; color: white; border-radius: 5px;")

        self.lbl_frame_stats.setText(f"Frame: {data['frame_count']}/{data['total_frames']}")
        self.lbl_det_stats.setText(f"Detected Frames: {data['det_frames']}")

    @Slot(dict)
    def update_cumulative(self, counts):
        self.list_cumulative.clear()
        for name, count in counts.items():
            self.list_cumulative.addItem(f"{name}: {count}")

    @Slot(int, int)
    def update_progress(self, current, total):
        if total > 0:
            val = int((current / total) * 100)
            self.progress_bar.setValue(val)

    def update_speed(self, val):
        if val <= 40:
            speed = 0.25 + (val / 40.0) * 0.75
        else:
            norm_val = (val - 40) / 60.0
            speed = 1.0 + 999.0 * (norm_val ** 3)
        self.lbl_speed.setText(f"Speed: {speed:.2f}x")
        
        if self.thread.isRunning():
            self.thread.set_speed(val)

    def prev_video(self):
        self.thread.request_prev()

    def next_video(self):
        self.thread.request_next()

    @Slot(dict)
    def collect_result(self, result):
        self.all_results.append(result)

    @Slot()
    def save_results(self):
        print("All processing finished. Saving results...")
        try:
            with open(OUTPUT_JSON, 'w') as f:
                json.dump(self.all_results, f, indent=2)
            print(f"Detailed report saved to {OUTPUT_JSON}")
            
            # Summary
            summary_results = []
            for r in self.all_results:
                s_item = copy.deepcopy(r)
                if 'detections' in s_item:
                    s_item['detection_count'] = len(s_item['detections'])
                    del s_item['detections']
                summary_results.append(s_item)
                
            with open(OUTPUT_JSON_SUMMARY, 'w') as f:
                json.dump(summary_results, f, indent=2)
            print(f"Summary report saved to {OUTPUT_JSON_SUMMARY}")
            
        except Exception as e:
            print(f"Error saving reports: {e}")
            
        self.close()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())