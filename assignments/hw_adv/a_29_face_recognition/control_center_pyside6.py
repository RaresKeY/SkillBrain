"""PySide6 control center for the A29 surveillance system."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import cv2
from PySide6.QtCore import QThread, Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from surveillance_core import (
    APP_DIR,
    DEFAULT_SETTINGS_PATH,
    EventLogger,
    PeopleRegistry,
    SettingsStore,
    SurveillanceEngine,
    SurveillanceSettings,
)


class SurveillanceWorker(QThread):
    frame_signal = Signal(QImage)
    stats_signal = Signal(dict)
    event_signal = Signal(str)
    error_signal = Signal(str)
    people_signal = Signal(list)
    stopped_signal = Signal()

    def __init__(self, settings: SurveillanceSettings):
        super().__init__()
        self.settings = settings
        self.registry = PeopleRegistry(
            registry_path=APP_DIR / self.settings.registry_path,
            known_people_dir=APP_DIR / self.settings.known_people_dir,
        )
        self.logger = EventLogger(logs_dir=APP_DIR / self.settings.logs_dir)
        self.engine = SurveillanceEngine(settings=self.settings, registry=self.registry, event_logger=self.logger)

    def _emit_event(self, event: dict) -> None:
        event_type = event.get("type", "EVENT")
        message = event.get("message", "")
        payload = event.get("payload", {})
        self.event_signal.emit(f"[{event_type}] {message} {payload}")

    def _emit_frame(self, frame_bgr, stats: dict) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.frame_signal.emit(qimg.copy())
        self.stats_signal.emit(stats)

    def run(self) -> None:
        try:
            self.engine.run(
                frame_callback=self._emit_frame,
                event_callback=self._emit_event,
                display=False,
                max_frames=0,
            )
        except Exception as err:
            self.error_signal.emit(str(err))
        finally:
            self.stopped_signal.emit()

    def stop(self) -> None:
        self.engine.request_stop()
        self.wait()

    def refresh_people(self) -> None:
        self.people_signal.emit(self.engine.list_people())

    def add_person_from_last_face(self, name: str) -> None:
        result = self.engine.add_person_from_latest_face(name)
        if result is None:
            self.event_signal.emit("[REGISTRY] No face available in the latest frame")
        else:
            self.event_signal.emit(f"[REGISTRY] Added person from live frame: {result}")
        self.refresh_people()

    def add_person_from_image_file(self, name: str, image_path: str) -> None:
        image = cv2.imread(image_path)
        if image is None:
            self.event_signal.emit(f"[REGISTRY] Failed to read image: {image_path}")
            return
        result = self.engine.add_person_from_image(name=name, image_bgr=image)
        if result is None:
            self.event_signal.emit("[REGISTRY] Failed to add person from image")
        else:
            self.event_signal.emit(f"[REGISTRY] Added person from image: {result}")
        self.refresh_people()

    def remove_person(self, person_id: str) -> None:
        ok = self.engine.remove_person(person_id)
        if ok:
            self.event_signal.emit(f"[REGISTRY] Removed person: {person_id}")
        else:
            self.event_signal.emit(f"[REGISTRY] Person not found: {person_id}")
        self.refresh_people()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings_store = SettingsStore(Path(DEFAULT_SETTINGS_PATH))
        self.settings = self.settings_store.load()
        self.worker: Optional[SurveillanceWorker] = None
        self.current_qimage: Optional[QImage] = None

        self.setWindowTitle("A29 Surveillance Control Center")
        self.resize(1500, 900)

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        self.video_label = QLabel("No video")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: #cccccc;")
        self.video_label.setMinimumSize(640, 400)

        self.metrics_label = QLabel("FPS: - | Motion: - | Faces: - | Alarm: -")
        self.metrics_label.setStyleSheet("background: #1e1e1e; color: #e5e5e5; padding: 6px;")

        self.events_text = QPlainTextEdit()
        self.events_text.setReadOnly(True)
        self.events_text.setMaximumBlockCount(600)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        left_layout.addWidget(self.video_label, stretch=1)
        left_layout.addWidget(self.metrics_label, stretch=0)
        left_layout.addWidget(self.events_text, stretch=0)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_content = QWidget()
        controls_layout = QVBoxLayout(controls_content)
        controls_layout.setContentsMargins(4, 4, 4, 4)
        controls_layout.setSpacing(10)

        controls_layout.addWidget(self._build_source_group())
        controls_layout.addWidget(self._build_detection_group())
        controls_layout.addWidget(self._build_alarm_group())
        controls_layout.addWidget(self._build_storage_group())
        controls_layout.addWidget(self._build_people_group())
        controls_layout.addWidget(self._build_actions_group())
        controls_layout.addStretch(1)

        controls_scroll.setWidget(controls_content)
        controls_scroll.setFixedWidth(460)

        root_layout.addWidget(left_panel, stretch=1)
        root_layout.addWidget(controls_scroll, stretch=0)

        self._populate_ui(self.settings)
        self.refresh_people_list()

    def _build_source_group(self) -> QWidget:
        group = QGroupBox("Source + Model")
        form = QFormLayout(group)

        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("0 for webcam, or /path/video.mp4")

        source_row = QWidget()
        source_layout = QHBoxLayout(source_row)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.addWidget(self.source_edit, stretch=1)
        self.browse_source_btn = QPushButton("Browse")
        self.browse_source_btn.clicked.connect(self.on_browse_source)
        source_layout.addWidget(self.browse_source_btn, stretch=0)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self._fill_model_combo()

        self.model_browse_btn = QPushButton("Browse Model")
        self.model_browse_btn.clicked.connect(self.on_browse_model)

        form.addRow("Source", source_row)
        form.addRow("Model", self.model_combo)
        form.addRow("", self.model_browse_btn)
        return group

    def _build_detection_group(self) -> QWidget:
        group = QGroupBox("Detection")
        form = QFormLayout(group)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.01)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 0.99)
        self.iou_spin.setSingleStep(0.01)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(160, 2048)
        self.imgsz_spin.setSingleStep(32)

        self.infer_every_spin = QSpinBox()
        self.infer_every_spin.setRange(1, 30)

        self.motion_enable_check = QCheckBox("Enable motion detection")
        self.motion_threshold_spin = QSpinBox()
        self.motion_threshold_spin.setRange(100, 500000)
        self.motion_threshold_spin.setSingleStep(100)

        self.min_contour_spin = QSpinBox()
        self.min_contour_spin.setRange(50, 200000)
        self.min_contour_spin.setSingleStep(50)

        self.face_threshold_spin = QDoubleSpinBox()
        self.face_threshold_spin.setRange(0.10, 0.99)
        self.face_threshold_spin.setSingleStep(0.01)

        form.addRow("Confidence", self.conf_spin)
        form.addRow("IoU", self.iou_spin)
        form.addRow("Image Size", self.imgsz_spin)
        form.addRow("Infer Every N", self.infer_every_spin)
        form.addRow("Motion", self.motion_enable_check)
        form.addRow("Motion Threshold", self.motion_threshold_spin)
        form.addRow("Min Contour Area", self.min_contour_spin)
        form.addRow("Face Match Threshold", self.face_threshold_spin)
        return group

    def _build_alarm_group(self) -> QWidget:
        group = QGroupBox("Alarm")
        form = QFormLayout(group)

        self.alarm_enable_check = QCheckBox("Enable alarm")
        self.alarm_motion_check = QCheckBox("Trigger on motion")
        self.alarm_unknown_check = QCheckBox("Trigger on unknown face")
        self.alarm_cooldown_spin = QDoubleSpinBox()
        self.alarm_cooldown_spin.setRange(0.1, 60.0)
        self.alarm_cooldown_spin.setSingleStep(0.1)

        form.addRow("Enable", self.alarm_enable_check)
        form.addRow("On Motion", self.alarm_motion_check)
        form.addRow("On Unknown", self.alarm_unknown_check)
        form.addRow("Cooldown (s)", self.alarm_cooldown_spin)
        return group

    def _build_storage_group(self) -> QWidget:
        group = QGroupBox("Storage + Runtime")
        form = QFormLayout(group)

        self.save_events_check = QCheckBox("Save event frames")
        self.save_unknown_faces_check = QCheckBox("Save unknown faces")
        self.loop_video_check = QCheckBox("Loop video files")
        self.snapshot_cooldown_spin = QDoubleSpinBox()
        self.snapshot_cooldown_spin.setRange(0.1, 60.0)
        self.snapshot_cooldown_spin.setSingleStep(0.1)

        form.addRow("Save Events", self.save_events_check)
        form.addRow("Save Unknown Faces", self.save_unknown_faces_check)
        form.addRow("Loop Video", self.loop_video_check)
        form.addRow("Snapshot Cooldown", self.snapshot_cooldown_spin)
        return group

    def _build_people_group(self) -> QWidget:
        group = QGroupBox("People Registry")
        layout = QVBoxLayout(group)

        self.people_list = QListWidget()
        layout.addWidget(self.people_list)

        self.person_name_edit = QLineEdit()
        self.person_name_edit.setPlaceholderText("Person name")
        layout.addWidget(self.person_name_edit)

        row1 = QWidget()
        row1_layout = QHBoxLayout(row1)
        row1_layout.setContentsMargins(0, 0, 0, 0)

        self.add_last_face_btn = QPushButton("Add From Last Face")
        self.add_last_face_btn.clicked.connect(self.on_add_from_last_face)
        row1_layout.addWidget(self.add_last_face_btn)

        self.add_from_image_btn = QPushButton("Add From Image")
        self.add_from_image_btn.clicked.connect(self.on_add_from_image)
        row1_layout.addWidget(self.add_from_image_btn)

        layout.addWidget(row1)

        row2 = QWidget()
        row2_layout = QHBoxLayout(row2)
        row2_layout.setContentsMargins(0, 0, 0, 0)

        self.remove_person_btn = QPushButton("Remove Selected")
        self.remove_person_btn.clicked.connect(self.on_remove_selected_person)
        row2_layout.addWidget(self.remove_person_btn)

        self.refresh_people_btn = QPushButton("Refresh")
        self.refresh_people_btn.clicked.connect(self.refresh_people_list)
        row2_layout.addWidget(self.refresh_people_btn)

        layout.addWidget(row2)
        return group

    def _build_actions_group(self) -> QWidget:
        group = QGroupBox("Actions")
        layout = QVBoxLayout(group)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_surveillance)
        layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_surveillance)
        layout.addWidget(self.stop_btn)

        self.save_settings_btn = QPushButton("Save Settings")
        self.save_settings_btn.clicked.connect(self.save_settings)
        layout.addWidget(self.save_settings_btn)

        self.reload_settings_btn = QPushButton("Reload Settings")
        self.reload_settings_btn.clicked.connect(self.reload_settings)
        layout.addWidget(self.reload_settings_btn)
        return group

    def _fill_model_combo(self) -> None:
        self.model_combo.clear()
        model_dir = APP_DIR / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        for model in sorted(model_dir.glob("*.pt")):
            self.model_combo.addItem(str(model.relative_to(APP_DIR)))

    def _populate_ui(self, settings: SurveillanceSettings) -> None:
        self.source_edit.setText(settings.source)
        self._fill_model_combo()
        self.model_combo.setCurrentText(settings.model_path)

        self.conf_spin.setValue(settings.confidence)
        self.iou_spin.setValue(settings.iou)
        self.imgsz_spin.setValue(settings.imgsz)
        self.infer_every_spin.setValue(settings.infer_every_n_frames)

        self.motion_enable_check.setChecked(settings.enable_motion)
        self.motion_threshold_spin.setValue(settings.motion_threshold)
        self.min_contour_spin.setValue(settings.min_contour_area)
        self.face_threshold_spin.setValue(settings.face_match_threshold)

        self.alarm_enable_check.setChecked(settings.enable_alarm)
        self.alarm_motion_check.setChecked(settings.alarm_on_motion)
        self.alarm_unknown_check.setChecked(settings.alarm_on_unknown_face)
        self.alarm_cooldown_spin.setValue(settings.alarm_cooldown_seconds)

        self.save_events_check.setChecked(settings.save_events)
        self.save_unknown_faces_check.setChecked(settings.save_unknown_faces)
        self.loop_video_check.setChecked(settings.loop_video)
        self.snapshot_cooldown_spin.setValue(settings.snapshot_cooldown_seconds)

    def _settings_from_ui(self) -> SurveillanceSettings:
        settings = SurveillanceSettings.from_dict(self.settings.to_dict())
        settings.source = self.source_edit.text().strip() or "0"
        settings.model_path = self.model_combo.currentText().strip() or settings.model_path
        settings.confidence = float(self.conf_spin.value())
        settings.iou = float(self.iou_spin.value())
        settings.imgsz = int(self.imgsz_spin.value())
        settings.infer_every_n_frames = int(self.infer_every_spin.value())
        settings.enable_motion = bool(self.motion_enable_check.isChecked())
        settings.motion_threshold = int(self.motion_threshold_spin.value())
        settings.min_contour_area = int(self.min_contour_spin.value())
        settings.face_match_threshold = float(self.face_threshold_spin.value())
        settings.enable_alarm = bool(self.alarm_enable_check.isChecked())
        settings.alarm_on_motion = bool(self.alarm_motion_check.isChecked())
        settings.alarm_on_unknown_face = bool(self.alarm_unknown_check.isChecked())
        settings.alarm_cooldown_seconds = float(self.alarm_cooldown_spin.value())
        settings.save_events = bool(self.save_events_check.isChecked())
        settings.save_unknown_faces = bool(self.save_unknown_faces_check.isChecked())
        settings.loop_video = bool(self.loop_video_check.isChecked())
        settings.snapshot_cooldown_seconds = float(self.snapshot_cooldown_spin.value())
        return settings

    def _registry(self) -> PeopleRegistry:
        settings = self._settings_from_ui()
        return PeopleRegistry(
            registry_path=APP_DIR / settings.registry_path,
            known_people_dir=APP_DIR / settings.known_people_dir,
        )

    def _append_event(self, text: str) -> None:
        self.events_text.appendPlainText(text)

    @Slot()
    def on_browse_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video Source",
            str(APP_DIR),
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)",
        )
        if path:
            self.source_edit.setText(path)

    @Slot()
    def on_browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Model",
            str(APP_DIR / "models"),
            "YOLO models (*.pt);;All files (*)",
        )
        if path:
            try:
                rel = str(Path(path).resolve().relative_to(APP_DIR))
            except ValueError:
                rel = path
            if self.model_combo.findText(rel) == -1:
                self.model_combo.addItem(rel)
            self.model_combo.setCurrentText(rel)

    @Slot()
    def start_surveillance(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self._append_event("[INFO] Surveillance is already running")
            return

        self.settings = self._settings_from_ui()
        self.worker = SurveillanceWorker(settings=self.settings)
        self.worker.frame_signal.connect(self.update_image)
        self.worker.stats_signal.connect(self.update_stats)
        self.worker.event_signal.connect(self._append_event)
        self.worker.error_signal.connect(self.on_worker_error)
        self.worker.people_signal.connect(self.on_people_signal)
        self.worker.stopped_signal.connect(self.on_worker_stopped)
        self.worker.start()
        self._append_event("[INFO] Surveillance started")

    @Slot()
    def stop_surveillance(self) -> None:
        if self.worker is None:
            return
        self.worker.stop()
        self.worker = None
        self._append_event("[INFO] Surveillance stopped")

    @Slot()
    def save_settings(self) -> None:
        self.settings = self._settings_from_ui()
        self.settings_store.save(self.settings)
        self._append_event(f"[INFO] Settings saved to {DEFAULT_SETTINGS_PATH}")

    @Slot()
    def reload_settings(self) -> None:
        self.settings = self.settings_store.load()
        self._populate_ui(self.settings)
        self._append_event("[INFO] Settings reloaded from disk")
        self.refresh_people_list()

    @Slot()
    def on_add_from_last_face(self) -> None:
        name = self.person_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing Name", "Enter a person name first.")
            return
        if self.worker is None or not self.worker.isRunning():
            QMessageBox.information(self, "Not Running", "Start surveillance first to capture a live face.")
            return
        self.worker.add_person_from_last_face(name)

    @Slot()
    def on_add_from_image(self) -> None:
        name = self.person_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing Name", "Enter a person name first.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Face Image",
            str(APP_DIR),
            "Image files (*.jpg *.jpeg *.png *.bmp);;All files (*)",
        )
        if not path:
            return

        if self.worker is not None and self.worker.isRunning():
            self.worker.add_person_from_image_file(name, path)
            return

        image = cv2.imread(path)
        if image is None:
            QMessageBox.warning(self, "Read Error", f"Could not load image: {path}")
            return
        registry = self._registry()
        result = registry.add_person(name=name, face_bgr=image)
        if result is None:
            QMessageBox.warning(self, "Registry Error", "Failed to add person from image.")
            return
        self._append_event(f"[REGISTRY] Added person from image: {result}")
        self.refresh_people_list()

    @Slot()
    def on_remove_selected_person(self) -> None:
        item = self.people_list.currentItem()
        if item is None:
            return
        person_id = item.data(Qt.ItemDataRole.UserRole)
        if not person_id:
            return

        if self.worker is not None and self.worker.isRunning():
            self.worker.remove_person(person_id)
            return

        registry = self._registry()
        ok = registry.remove_person(person_id)
        if ok:
            self._append_event(f"[REGISTRY] Removed person: {person_id}")
        else:
            self._append_event(f"[REGISTRY] Person not found: {person_id}")
        self.refresh_people_list()

    @Slot()
    def refresh_people_list(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.refresh_people()
            return
        registry = self._registry()
        self.on_people_signal(registry.list_people())

    @Slot(list)
    def on_people_signal(self, people: list) -> None:
        self.people_list.clear()
        for person in people:
            label = f"{person['name']} ({person.get('samples_count', 0)} samples)"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, person["person_id"])
            self.people_list.addItem(item)

    @Slot(QImage)
    def update_image(self, image: QImage) -> None:
        self.current_qimage = image
        self.render_current_frame()

    @Slot(dict)
    def update_stats(self, stats: dict) -> None:
        metrics = (
            f"FPS: {stats.get('fps', '-')}"
            f" | Motion: {stats.get('motion_pixels', '-')}"
            f" | Persons: {stats.get('person_count', '-')}"
            f" | Faces: {stats.get('face_count', '-')}"
            f" | Known: {stats.get('known_faces', '-')}"
            f" | Unknown: {stats.get('unknown_faces', '-')}"
            f" | Alarm: {stats.get('alarm_active', False)}"
        )
        self.metrics_label.setText(metrics)

    @Slot(str)
    def on_worker_error(self, message: str) -> None:
        self._append_event(f"[ERROR] {message}")
        QMessageBox.critical(self, "Worker Error", message)

    @Slot()
    def on_worker_stopped(self) -> None:
        self._append_event("[INFO] Worker thread exited")

    def render_current_frame(self) -> None:
        if self.current_qimage is None:
            return
        pixmap = QPixmap.fromImage(self.current_qimage).scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.render_current_frame()

    def closeEvent(self, event) -> None:
        self.stop_surveillance()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
