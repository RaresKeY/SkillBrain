import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import cv2
import numpy as np
import yt_dlp
from PySide6.QtCore import QThread, Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import yolo_feed_config as cfg

try:
    from ultralytics import YOLO
except Exception as import_exc:  # pragma: no cover
    YOLO = None
    YOLO_IMPORT_ERROR = import_exc
else:
    YOLO_IMPORT_ERROR = None

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"

MODEL_DOWNLOAD_URLS = {
    "yolov5m.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt",
    "yolov5l.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt",
    "yolov5x.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt",
    "yolo11m.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
    "yolo11l.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt",
}


@dataclass
class FeedSettings:
    source: str
    model_name: str
    conf: float
    imgsz: int
    infer_every_n: int
    draw_classes: list[str]
    ydl_format: str
    save_detections: bool
    save_dir: str
    save_min_interval_s: float
    window_title: str
    window_width: int
    window_height: int


def default_settings() -> FeedSettings:
    return FeedSettings(
        source=cfg.SOURCE_URL,
        model_name=cfg.MODEL_NAME,
        conf=cfg.CONF_THRESHOLD,
        imgsz=cfg.INFERENCE_IMAGE_SIZE,
        infer_every_n=cfg.INFER_EVERY_N_FRAMES,
        draw_classes=list(cfg.DRAW_CLASSES),
        ydl_format=cfg.YDL_FORMAT,
        save_detections=cfg.SAVE_DETECTIONS,
        save_dir=cfg.SAVE_DIR,
        save_min_interval_s=cfg.SAVE_MIN_INTERVAL_SECONDS,
        window_title=cfg.WINDOW_TITLE,
        window_width=cfg.WINDOW_WIDTH,
        window_height=cfg.WINDOW_HEIGHT,
    )


def write_config_file(settings: FeedSettings) -> None:
    """Persist current settings to yolo_feed_config.py."""
    config_path = Path(cfg.__file__)
    model_options = list(dict.fromkeys([*cfg.MODEL_OPTIONS, settings.model_name]))

    content = f'''"""Config for PySide6 YouTube feed with YOLO overlay."""

# Stream source (YouTube live/video URL)
SOURCE_URL = {settings.source!r}

# Recommended model for this wide outdoor scene (good small-object recall vs speed)
MODEL_NAME = {settings.model_name!r}

# Available model options shown in the UI
MODEL_OPTIONS = {model_options!r}

# Detection and inference settings
CONF_THRESHOLD = {settings.conf!r}
INFERENCE_IMAGE_SIZE = {settings.imgsz!r}
INFER_EVERY_N_FRAMES = {settings.infer_every_n!r}

# Draw only selected COCO classes (empty list means all classes)
DRAW_CLASSES = {settings.draw_classes!r}

# yt-dlp stream format selector
YDL_FORMAT = {settings.ydl_format!r}

# Save detection snapshots
SAVE_DETECTIONS = {settings.save_detections!r}
SAVE_DIR = {settings.save_dir!r}
SAVE_MIN_INTERVAL_SECONDS = {settings.save_min_interval_s!r}

# UI defaults
WINDOW_TITLE = {settings.window_title!r}
WINDOW_WIDTH = {settings.window_width!r}
WINDOW_HEIGHT = {settings.window_height!r}
'''
    config_path.write_text(content, encoding="utf-8")


def resolve_model_path(model_name: str, status_cb=None) -> Path:
    """Ensure model exists under local ./models folder, downloading if needed."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = Path(model_name)
    if model_path.is_absolute() and model_path.exists():
        return model_path

    local_model = MODELS_DIR / model_path.name
    if local_model.exists():
        return local_model

    url = MODEL_DOWNLOAD_URLS.get(model_path.name)
    if not url:
        raise RuntimeError(
            f"Model '{model_name}' not found in local models folder and no download URL configured."
        )

    if status_cb is not None:
        status_cb(f"Downloading model to {local_model} ...")
    urlretrieve(url, str(local_model))
    return local_model


class VideoThread(QThread):
    change_pixmap_signal = Signal(QImage)
    status_signal = Signal(str)

    def __init__(self, settings: FeedSettings):
        super().__init__()
        self.settings = settings
        self._run_flag = True

        self.model = None
        self.names = {}
        self.allowed_class_ids: Optional[set[int]] = None
        self.last_save_ts = 0.0

    def stop(self):
        self._run_flag = False
        self.wait()

    def _resolve_stream_url(self) -> str:
        ydl_opts = {
            "format": self.settings.ydl_format,
            "quiet": True,
            "no_warnings": True,
            "live_from_start": False,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(self.settings.source, download=False)

        if "url" in info:
            return info["url"]

        for fmt in info.get("formats", []):
            if fmt.get("url"):
                return fmt["url"]

        raise RuntimeError("Could not resolve stream URL from source")

    def _load_model(self):
        if YOLO is None:
            raise RuntimeError(f"Failed to import ultralytics: {YOLO_IMPORT_ERROR}")

        model_path = resolve_model_path(self.settings.model_name, status_cb=self.status_signal.emit)
        self.status_signal.emit(f"Loading model: {model_path.name} ...")
        self.model = YOLO(str(model_path))
        self.names = self.model.names if hasattr(self.model, "names") else {}

        draw_classes = [c.strip() for c in self.settings.draw_classes if c.strip()]
        if draw_classes:
            allowed = set()
            for cid, cname in self.names.items():
                if cname in draw_classes:
                    allowed.add(int(cid))
            self.allowed_class_ids = allowed
        else:
            self.allowed_class_ids = None

    def _draw_detections(self, frame_bgr: np.ndarray, result) -> tuple[np.ndarray, int]:
        out = frame_bgr.copy()
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return out, 0

        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
        clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []

        count = 0
        for box, conf, cls_id in zip(xyxy, confs, clss):
            if self.allowed_class_ids is not None and cls_id not in self.allowed_class_ids:
                continue

            x1, y1, x2, y2 = map(int, box.tolist())
            label = self.names.get(cls_id, str(cls_id))
            text = f"{label} {conf:.2f}"
            color = (0, 220, 0) if label == "person" else (0, 165, 255)

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                out,
                text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
            count += 1

        return out, count

    def _maybe_save_detection_image(self, annotated_bgr: np.ndarray, det_count: int) -> None:
        if not self.settings.save_detections or det_count <= 0:
            return

        now = datetime.now()
        now_ts = now.timestamp()
        if now_ts - self.last_save_ts < self.settings.save_min_interval_s:
            return

        out_dir = SCRIPT_DIR / self.settings.save_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = now.strftime("det_%Y%m%d_%H%M%S_%f")[:-3] + ".jpg"
        out_path = out_dir / filename
        cv2.imwrite(str(out_path), annotated_bgr)
        self.last_save_ts = now_ts
        self.status_signal.emit(f"Saved detection frame: {out_path}")

    def run(self):
        try:
            self._load_model()
            stream_url = self._resolve_stream_url()
            self.status_signal.emit("Model loaded. Connecting to stream...")
        except Exception as e:
            self.status_signal.emit(f"Startup error: {e}")
            return

        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(stream_url)

        if not cap.isOpened():
            self.status_signal.emit("Error opening stream")
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 30.0
        delay_ms = int(1000 / fps)

        frame_idx = 0
        last_annotated_rgb = None

        while self._run_flag:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                self.status_signal.emit("Stream read failed, reconnecting...")
                cap.release()
                self.msleep(2000)
                cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(stream_url)
                continue

            frame_idx += 1

            if frame_idx % self.settings.infer_every_n == 0:
                try:
                    results = self.model.predict(
                        source=frame_bgr,
                        conf=self.settings.conf,
                        imgsz=self.settings.imgsz,
                        verbose=False,
                    )
                    annotated_bgr, det_count = self._draw_detections(frame_bgr, results[0])
                    self._maybe_save_detection_image(annotated_bgr, det_count)
                    last_annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                except Exception as infer_err:
                    self.status_signal.emit(f"Inference error: {infer_err}")
                    last_annotated_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            elif last_annotated_rgb is None:
                last_annotated_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            h, w, ch = last_annotated_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(last_annotated_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.change_pixmap_signal.emit(qimg.copy())
            self.msleep(max(1, delay_ms))

        cap.release()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = default_settings()

        self.setWindowTitle(self.settings.window_title)
        self.resize(self.settings.window_width, self.settings.window_height)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        self.image_label.setMinimumSize(1, 1)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("padding: 6px; color: #d8d8d8; background: #222;")

        self.current_qimage = None
        self.thread: Optional[VideoThread] = None

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(0)
        video_layout.addWidget(self.image_label, stretch=1)
        video_layout.addWidget(self.status_label, stretch=0)

        panel = self._build_control_panel()
        panel.setFixedWidth(360)

        root_layout.addWidget(video_container, stretch=1)
        root_layout.addWidget(panel, stretch=0)

        self._populate_ui_from_settings(self.settings)
        self.restart_stream()

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        form = QFormLayout()

        self.source_edit = QLineEdit()

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        for model in cfg.MODEL_OPTIONS:
            self.model_combo.addItem(model)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.01)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(160, 2048)
        self.imgsz_spin.setSingleStep(32)

        self.infer_every_spin = QSpinBox()
        self.infer_every_spin.setRange(1, 20)

        self.classes_edit = QLineEdit()

        self.ydl_format_edit = QLineEdit()

        self.save_checkbox = QCheckBox("Enable save")
        self.save_dir_edit = QLineEdit()
        self.save_browse_btn = QPushButton("Browse")
        self.save_browse_btn.clicked.connect(self.browse_save_dir)

        save_dir_row = QWidget()
        save_dir_layout = QHBoxLayout(save_dir_row)
        save_dir_layout.setContentsMargins(0, 0, 0, 0)
        save_dir_layout.addWidget(self.save_dir_edit, stretch=1)
        save_dir_layout.addWidget(self.save_browse_btn, stretch=0)

        self.save_interval_spin = QDoubleSpinBox()
        self.save_interval_spin.setRange(0.10, 60.0)
        self.save_interval_spin.setSingleStep(0.10)

        self.window_title_edit = QLineEdit()
        self.window_width_spin = QSpinBox()
        self.window_width_spin.setRange(640, 3840)
        self.window_height_spin = QSpinBox()
        self.window_height_spin.setRange(360, 2160)

        form.addRow("Source URL", self.source_edit)
        form.addRow("YOLO Model", self.model_combo)
        form.addRow("Confidence", self.conf_spin)
        form.addRow("Image Size", self.imgsz_spin)
        form.addRow("Infer Every N", self.infer_every_spin)
        form.addRow("Class Filter", self.classes_edit)
        form.addRow("yt-dlp Format", self.ydl_format_edit)
        form.addRow("Save Detections", self.save_checkbox)
        form.addRow("Save Directory", save_dir_row)
        form.addRow("Save Min Interval", self.save_interval_spin)
        form.addRow("Window Title", self.window_title_edit)
        form.addRow("Window Width", self.window_width_spin)
        form.addRow("Window Height", self.window_height_spin)

        layout.addLayout(form)

        self.apply_btn = QPushButton("Apply + Restart")
        self.apply_btn.clicked.connect(self.apply_and_restart)

        self.save_cfg_btn = QPushButton("Save Settings To Config")
        self.save_cfg_btn.clicked.connect(self.save_config_only)

        self.stop_btn = QPushButton("Stop Stream")
        self.stop_btn.clicked.connect(self.stop_stream)

        self.start_btn = QPushButton("Start Stream")
        self.start_btn.clicked.connect(self.restart_stream)

        layout.addWidget(self.apply_btn)
        layout.addWidget(self.save_cfg_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.start_btn)
        layout.addStretch(1)
        return panel

    def _populate_ui_from_settings(self, s: FeedSettings) -> None:
        self.source_edit.setText(s.source)
        self.model_combo.setCurrentText(s.model_name)
        self.conf_spin.setValue(float(s.conf))
        self.imgsz_spin.setValue(int(s.imgsz))
        self.infer_every_spin.setValue(int(s.infer_every_n))
        self.classes_edit.setText(", ".join(s.draw_classes))
        self.ydl_format_edit.setText(s.ydl_format)
        self.save_checkbox.setChecked(bool(s.save_detections))
        self.save_dir_edit.setText(s.save_dir)
        self.save_interval_spin.setValue(float(s.save_min_interval_s))
        self.window_title_edit.setText(s.window_title)
        self.window_width_spin.setValue(int(s.window_width))
        self.window_height_spin.setValue(int(s.window_height))

    def _settings_from_ui(self) -> FeedSettings:
        classes = [c.strip() for c in self.classes_edit.text().split(",") if c.strip()]
        return FeedSettings(
            source=self.source_edit.text().strip(),
            model_name=self.model_combo.currentText().strip(),
            conf=float(self.conf_spin.value()),
            imgsz=int(self.imgsz_spin.value()),
            infer_every_n=int(self.infer_every_spin.value()),
            draw_classes=classes,
            ydl_format=self.ydl_format_edit.text().strip(),
            save_detections=bool(self.save_checkbox.isChecked()),
            save_dir=self.save_dir_edit.text().strip() or "detections",
            save_min_interval_s=float(self.save_interval_spin.value()),
            window_title=self.window_title_edit.text().strip() or cfg.WINDOW_TITLE,
            window_width=int(self.window_width_spin.value()),
            window_height=int(self.window_height_spin.value()),
        )

    def browse_save_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if folder:
            self.save_dir_edit.setText(folder)

    def apply_and_restart(self) -> None:
        self.settings = self._settings_from_ui()
        self.setWindowTitle(self.settings.window_title)
        self.resize(self.settings.window_width, self.settings.window_height)
        write_config_file(self.settings)
        self.restart_stream()
        self.set_status("Settings applied, stream restarted, config synced")

    def save_config_only(self) -> None:
        self.settings = self._settings_from_ui()
        write_config_file(self.settings)
        self.set_status("Config file updated")

    def stop_stream(self) -> None:
        if self.thread is not None:
            self.thread.stop()
            self.thread = None
            self.set_status("Stream stopped")

    def restart_stream(self) -> None:
        self.stop_stream()
        self.settings = self._settings_from_ui()
        self.thread = VideoThread(self.settings)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.status_signal.connect(self.set_status)
        self.thread.start()

    @Slot(QImage)
    def update_image(self, qt_img):
        self.current_qimage = qt_img
        self.display_image()

    @Slot(str)
    def set_status(self, text: str):
        self.status_label.setText(text)

    def display_image(self):
        if self.current_qimage:
            scaled_pixmap = QPixmap.fromImage(self.current_qimage).scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.display_image()

    def closeEvent(self, event):
        self.stop_stream()
        event.accept()


if __name__ == "__main__":
    # Make all relative paths resolve to this assignment folder.
    import os

    os.chdir(SCRIPT_DIR)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
