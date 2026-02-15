"""Shared surveillance engine for CLI and PySide6 control applications.

This implementation adapts patterns from:
- a_28_oos_computer_vision/person_detect (YOLO person+face flow)
- a_28_opencv/opencv_surveillance_project (motion event logic)
- a_39_40_yolo_on_feed (threaded UI integration patterns)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
import json
import re
import shutil
import sys
import threading
import time
from urllib.request import urlretrieve

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


APP_DIR = Path(__file__).resolve().parent
DEFAULT_SETTINGS_PATH = APP_DIR / "config" / "settings.json"

MODEL_DOWNLOAD_URLS = {
    "yolo11m.pt": [
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
    ],
    "yolov8n-face.pt": [
        "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt",
    ],
}

LOCAL_MODEL_COPY_CANDIDATES = {
    "yolov8x_person_face.pt": [
        APP_DIR.parent / "a_28_oos_computer_vision" / "person_detect" / "models" / "yolov8x_person_face.pt",
        APP_DIR.parent / "a_28_oos_computer_vision" / "fire_detect" / "models" / "yolov8x_person_face.pt",
    ],
    "yolo11m.pt": [
        APP_DIR.parent / "a_39_40_yolo_on_feed" / "models" / "yolo11m.pt",
    ],
}


@dataclass
class SurveillanceSettings:
    source: str = "0"
    window_title: str = "A29 Surveillance System"
    model_path: str = "models/yolov8x_person_face.pt"
    fallback_model_path: str = "models/yolo11m.pt"
    confidence: float = 0.40
    iou: float = 0.45
    imgsz: int = 960
    infer_every_n_frames: int = 1
    enable_motion: bool = False
    motion_threshold: int = 2000
    min_contour_area: int = 900
    blur_kernel_size: int = 21
    enable_alarm: bool = True
    alarm_on_motion: bool = False
    alarm_on_unknown_face: bool = True
    alarm_cooldown_seconds: float = 3.0
    snapshot_cooldown_seconds: float = 1.0
    save_events: bool = True
    save_unknown_faces: bool = True
    loop_video: bool = False
    face_match_threshold: float = 0.72
    events_dir: str = "output/events"
    unknown_face_dir: str = "output/unknown_faces"
    logs_dir: str = "logs"
    known_people_dir: str = "data/known_people"
    registry_path: str = "data/people_registry.json"
    auto_download_missing_model: bool = True
    face_model_download_url: str = (
        "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt"
    )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SurveillanceSettings":
        known = {f.name for f in fields(cls)}
        safe_payload = {k: v for k, v in payload.items() if k in known}
        return cls(**safe_payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SettingsStore:
    def __init__(self, path: Path = DEFAULT_SETTINGS_PATH):
        self.path = path

    def load(self) -> SurveillanceSettings:
        if not self.path.exists():
            settings = SurveillanceSettings()
            self.save(settings)
            return settings

        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return SurveillanceSettings.from_dict(payload)

    def save(self, settings: SurveillanceSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(settings.to_dict(), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )


@dataclass
class PersonRecord:
    person_id: str
    name: str
    created_at: str
    samples: list[str] = field(default_factory=list)
    descriptors: list[list[float]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PersonRecord":
        return cls(
            person_id=payload["person_id"],
            name=payload["name"],
            created_at=payload.get("created_at", datetime.now().isoformat(timespec="seconds")),
            samples=list(payload.get("samples", [])),
            descriptors=list(payload.get("descriptors", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "name": self.name,
            "created_at": self.created_at,
            "samples": self.samples,
            "descriptors": self.descriptors,
        }


class PeopleRegistry:
    def __init__(self, registry_path: Path, known_people_dir: Path):
        self.registry_path = registry_path
        self.known_people_dir = known_people_dir
        self._lock = threading.Lock()
        self.records: list[PersonRecord] = []
        self._load()

    def _load(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.known_people_dir.mkdir(parents=True, exist_ok=True)

        if not self.registry_path.exists():
            self._save()
            return

        payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        people = payload.get("people", [])
        self.records = [PersonRecord.from_dict(item) for item in people]

    def _save(self) -> None:
        payload = {
            "version": 1,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "people": [rec.to_dict() for rec in self.records],
        }
        self.registry_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    @staticmethod
    def _slug(value: str) -> str:
        clean = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
        return clean or "person"

    @staticmethod
    def _extract_face_descriptor(face_bgr: np.ndarray) -> Optional[np.ndarray]:
        if face_bgr is None or face_bgr.size == 0:
            return None

        resized = cv2.resize(face_bgr, (96, 96), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        intensity_hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten().astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        orient_hist, _ = np.histogram(ang, bins=18, range=(0, 360), weights=mag)
        orient_hist = orient_hist.astype(np.float32)

        descriptor = np.concatenate([intensity_hist, orient_hist], axis=0)
        norm = np.linalg.norm(descriptor)
        if norm <= 1e-8:
            return None
        return descriptor / norm

    @staticmethod
    def extract_face_descriptor(face_bgr: np.ndarray) -> Optional[np.ndarray]:
        return PeopleRegistry._extract_face_descriptor(face_bgr)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-8:
            return 0.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return PeopleRegistry._cosine_similarity(a, b)

    def list_people(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "person_id": rec.person_id,
                    "name": rec.name,
                    "created_at": rec.created_at,
                    "samples_count": len(rec.samples),
                }
                for rec in self.records
            ]

    def add_person(self, name: str, face_bgr: np.ndarray) -> Optional[dict[str, Any]]:
        descriptor = self._extract_face_descriptor(face_bgr)
        if descriptor is None:
            return None

        with self._lock:
            now = datetime.now()
            target = None
            for rec in self.records:
                if rec.name.strip().lower() == name.strip().lower():
                    target = rec
                    break

            if target is None:
                person_id = f"{self._slug(name)}_{now.strftime('%Y%m%d_%H%M%S')}"
                target = PersonRecord(
                    person_id=person_id,
                    name=name.strip(),
                    created_at=now.isoformat(timespec="seconds"),
                )
                self.records.append(target)

            sample_idx = len(target.samples) + 1
            sample_file = self.known_people_dir / f"{target.person_id}_{sample_idx:03d}.jpg"
            cv2.imwrite(str(sample_file), face_bgr)

            try:
                relative_sample = str(sample_file.relative_to(APP_DIR))
            except ValueError:
                relative_sample = str(sample_file)

            target.samples.append(relative_sample)
            target.descriptors.append(descriptor.astype(float).tolist())
            self._save()
            return {
                "person_id": target.person_id,
                "name": target.name,
                "sample_path": relative_sample,
                "samples_count": len(target.samples),
            }

    def remove_person(self, person_id: str) -> bool:
        with self._lock:
            target = None
            for rec in self.records:
                if rec.person_id == person_id:
                    target = rec
                    break

            if target is None:
                return False

            for sample in target.samples:
                sample_path = APP_DIR / sample
                if sample_path.exists():
                    sample_path.unlink()

            self.records.remove(target)
            self._save()
            return True

    def match(self, face_bgr: np.ndarray, threshold: float) -> Optional[dict[str, Any]]:
        descriptor = self._extract_face_descriptor(face_bgr)
        if descriptor is None:
            return None

        best_person: Optional[PersonRecord] = None
        best_score = -1.0

        with self._lock:
            for rec in self.records:
                for stored in rec.descriptors:
                    stored_arr = np.asarray(stored, dtype=np.float32)
                    score = self._cosine_similarity(descriptor, stored_arr)
                    if score > best_score:
                        best_score = score
                        best_person = rec

        if best_person is None or best_score < threshold:
            return None

        return {
            "person_id": best_person.person_id,
            "name": best_person.name,
            "score": round(best_score, 4),
        }


class EventLogger:
    def __init__(self, logs_dir: Path):
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_txt_path = self.logs_dir / "surveillance.log"
        self.log_jsonl_path = self.logs_dir / "events.jsonl"
        self._lock = threading.Lock()

    def log(self, event_type: str, message: str, payload: Optional[dict[str, Any]] = None) -> None:
        now = datetime.now()
        time_text = now.strftime("%Y-%m-%d %H:%M:%S")
        json_event = {
            "timestamp": now.isoformat(timespec="seconds"),
            "event_type": event_type,
            "message": message,
            "payload": payload or {},
        }

        line = f"[{time_text}] {event_type}: {message}"
        if payload:
            line += f" | payload={payload}"
        line += "\n"

        with self._lock:
            with self.log_txt_path.open("a", encoding="utf-8") as txt_f:
                txt_f.write(line)
            with self.log_jsonl_path.open("a", encoding="utf-8") as jsonl_f:
                jsonl_f.write(json.dumps(json_event, ensure_ascii=True) + "\n")


@dataclass
class MotionResult:
    detected: bool
    pixel_count: int
    boxes: list[tuple[int, int, int, int]]
    mask: np.ndarray
    current_gray: np.ndarray


class SurveillanceEngine:
    def __init__(
        self,
        settings: SurveillanceSettings,
        registry: PeopleRegistry,
        event_logger: EventLogger,
    ):
        self.settings = settings
        self.registry = registry
        self.event_logger = event_logger

        self._stop_requested = False
        self._model = None
        self._model_names: dict[int, str] = {}
        self._person_class_ids: set[int] = set()
        self._face_class_ids: set[int] = set()
        self._last_infer_detections: list[dict[str, Any]] = []
        self._last_event_ts: dict[str, float] = {}
        self._latest_face_crops: list[np.ndarray] = []
        self._latest_faces_lock = threading.Lock()
        self._source_is_file = False
        self._unknown_saved_descriptors: list[np.ndarray] = []
        self._unknown_cache_loaded = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def reset_stop_flag(self) -> None:
        self._stop_requested = False

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return APP_DIR / path

    def _resolve_model_path(self) -> Path:
        candidates = [
            self.settings.model_path,
            self.settings.fallback_model_path,
            "models/yolov8x_person_face.pt",
            "models/yolov8n-face.pt",
            "models/yolo11m.pt",
            "models/yolov5lu.pt",
        ]
        for candidate in candidates:
            model_path = self._resolve_path(candidate)
            if model_path.exists() or self._ensure_model_available(model_path):
                return model_path
        raise FileNotFoundError("No model file found in configured model paths.")

    def _ensure_model_available(self, model_path: Path) -> bool:
        if model_path.exists():
            return True
        if not self.settings.auto_download_missing_model:
            return False

        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_name = model_path.name

        for local_src in LOCAL_MODEL_COPY_CANDIDATES.get(model_name, []):
            if local_src.exists():
                try:
                    shutil.copy2(local_src, model_path)
                    self.event_logger.log(
                        "MODEL",
                        "Copied missing model from local source",
                        {"source": str(local_src), "destination": str(model_path)},
                    )
                    return True
                except Exception as err:
                    self.event_logger.log(
                        "WARNING",
                        "Failed local model copy",
                        {"source": str(local_src), "error": str(err)},
                    )

        download_urls = list(MODEL_DOWNLOAD_URLS.get(model_name, []))
        if model_name in {"yolov8x_person_face.pt", "yolov8n-face.pt"} and self.settings.face_model_download_url.strip():
            custom_face_url = self.settings.face_model_download_url.strip()
            if custom_face_url not in download_urls:
                download_urls.insert(0, custom_face_url)

        for url in download_urls:
            try:
                self.event_logger.log(
                    "MODEL",
                    "Downloading missing model",
                    {"model": model_name, "url": url},
                )
                urlretrieve(url, str(model_path))
                self.event_logger.log(
                    "MODEL",
                    "Model download completed",
                    {"model": model_name, "path": str(model_path)},
                )
                return True
            except Exception as err:
                self.event_logger.log(
                    "WARNING",
                    "Model download attempt failed",
                    {"model": model_name, "url": url, "error": str(err)},
                )

        return model_path.exists()

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed. Install dependencies from requirements.txt.")

        model_path = self._resolve_model_path()
        self._model = YOLO(str(model_path))

        if torch is not None:
            try:
                device = 0 if torch.cuda.is_available() else "cpu"
                self._model.to(device)
            except Exception:
                pass

        names = getattr(self._model, "names", {})
        self._model_names = {int(k): str(v) for k, v in names.items()}
        self._person_class_ids = {
            cid for cid, name in self._model_names.items() if name.strip().lower() == "person"
        }
        self._face_class_ids = {
            cid for cid, name in self._model_names.items() if "face" in name.strip().lower()
        }

        self.event_logger.log(
            "INFO",
            "Model loaded",
            {"model_path": str(model_path), "face_classes": sorted(self._face_class_ids)},
        )

    def _resolve_source(self) -> Any:
        source_text = str(self.settings.source).strip()
        as_path = self._resolve_path(source_text)
        if as_path.exists():
            self._source_is_file = True
            return str(as_path)

        if source_text.lower().startswith("camera:"):
            self._source_is_file = False
            return int(source_text.split(":", 1)[1])

        if source_text.isdigit():
            self._source_is_file = False
            return int(source_text)

        self._source_is_file = False
        return source_text

    def _open_capture(self) -> cv2.VideoCapture:
        source = self._resolve_source()
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source: {self.settings.source}")
        return cap

    def _blur_kernel_size(self) -> int:
        kernel = int(self.settings.blur_kernel_size)
        if kernel < 3:
            return 3
        if kernel % 2 == 0:
            kernel += 1
        return kernel

    def _detect_motion(self, previous_gray: Optional[np.ndarray], frame_bgr: np.ndarray) -> MotionResult:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self._blur_kernel_size(), self._blur_kernel_size()), 0)

        if not self.settings.enable_motion:
            return MotionResult(
                detected=False,
                pixel_count=0,
                boxes=[],
                mask=np.zeros_like(gray),
                current_gray=gray,
            )

        if previous_gray is None:
            return MotionResult(
                detected=False,
                pixel_count=0,
                boxes=[],
                mask=np.zeros_like(gray),
                current_gray=gray,
            )

        frame_delta = cv2.absdiff(previous_gray, gray)
        _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[tuple[int, int, int, int]] = []
        pixel_count = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.settings.min_contour_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))
            pixel_count += int(area)

        detected = self.settings.enable_motion and pixel_count >= self.settings.motion_threshold
        return MotionResult(
            detected=detected,
            pixel_count=pixel_count,
            boxes=boxes,
            mask=thresh,
            current_gray=gray,
        )

    def _run_inference(self, frame_bgr: np.ndarray) -> list[dict[str, Any]]:
        self._load_model()

        try:
            results = self._model.predict(
                source=frame_bgr,
                conf=float(self.settings.confidence),
                iou=float(self.settings.iou),
                imgsz=int(self.settings.imgsz),
                verbose=False,
            )
        except Exception as err:
            self.event_logger.log("ERROR", "Inference failed", {"error": str(err)})
            return []

        if not results:
            return []

        detections: list[dict[str, Any]] = []
        boxes = getattr(results[0], "boxes", None)
        if boxes is not None and boxes.xyxy is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))
            clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)

            for box, conf, cls_id in zip(xyxy, confs, clss):
                x1, y1, x2, y2 = map(int, box.tolist())
                label = self._model_names.get(int(cls_id), str(cls_id)).strip().lower()
                is_person = int(cls_id) in self._person_class_ids or label == "person"
                is_face = int(cls_id) in self._face_class_ids or label == "face" or "face" in label
                detections.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "conf": float(conf),
                        "cls_id": int(cls_id),
                        "label": label,
                        "is_person": is_person,
                        "is_face": is_face,
                        "source": "yolo",
                    }
                )

        return detections

    @staticmethod
    def _safe_crop(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = bbox
        h, w = frame_bgr.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 - x1 < 20 or y2 - y1 < 20:
            return None
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def _analyze_faces(
        self,
        frame_bgr: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int, int]:
        face_infos: list[dict[str, Any]] = []
        known_count = 0
        unknown_count = 0
        crops_for_ui: list[np.ndarray] = []

        for det in detections:
            if not det["is_face"]:
                continue

            crop = self._safe_crop(frame_bgr, det["bbox"])
            if crop is None:
                continue

            match = self.registry.match(crop, threshold=float(self.settings.face_match_threshold))
            if match is None:
                label = "Unknown"
                score = 0.0
                known = False
                unknown_count += 1
            else:
                label = match["name"]
                score = float(match["score"])
                known = True
                known_count += 1

            face_infos.append(
                {
                    "bbox": det["bbox"],
                    "name": label,
                    "score": score,
                    "known": known,
                }
            )
            crops_for_ui.append(crop.copy())

        with self._latest_faces_lock:
            self._latest_face_crops = crops_for_ui

        return face_infos, known_count, unknown_count

    def _cooldown_ready(self, key: str, seconds: float, now_ts: float) -> bool:
        last_ts = self._last_event_ts.get(key, 0.0)
        if now_ts - last_ts < seconds:
            return False
        self._last_event_ts[key] = now_ts
        return True

    def _save_frame(self, frame_bgr: np.ndarray, out_dir: str, prefix: str) -> str:
        path = self._resolve_path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.jpg"
        full_path = path / filename
        cv2.imwrite(str(full_path), frame_bgr)
        try:
            return str(full_path.relative_to(APP_DIR))
        except ValueError:
            return str(full_path)

    def _load_unknown_descriptor_cache(self) -> None:
        if self._unknown_cache_loaded:
            return

        unknown_dir = self._resolve_path(self.settings.unknown_face_dir)
        unknown_dir.mkdir(parents=True, exist_ok=True)
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for img_path in sorted(unknown_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in image_exts:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            descriptor = PeopleRegistry.extract_face_descriptor(img)
            if descriptor is not None:
                self._unknown_saved_descriptors.append(descriptor)

        self._unknown_cache_loaded = True

    def _should_save_unknown_face(self, face_crop: np.ndarray, similarity_threshold: float = 0.97) -> bool:
        descriptor = PeopleRegistry.extract_face_descriptor(face_crop)
        if descriptor is None:
            return False

        self._load_unknown_descriptor_cache()
        if self._unknown_saved_descriptors:
            similarities = [
                PeopleRegistry.cosine_similarity(descriptor, saved_desc)
                for saved_desc in self._unknown_saved_descriptors
            ]
            if similarities and max(similarities) >= similarity_threshold:
                return False

        self._unknown_saved_descriptors.append(descriptor)
        return True

    def _draw_annotations(
        self,
        frame_bgr: np.ndarray,
        detections: list[dict[str, Any]],
        face_infos: list[dict[str, Any]],
        motion: MotionResult,
        alarm_active: bool,
        fps: float,
    ) -> np.ndarray:
        out = frame_bgr.copy()

        for det in detections:
            if det["is_face"]:
                continue
            if not det["is_person"]:
                continue
            x1, y1, x2, y2 = det["bbox"]
            conf = det["conf"]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(
                out,
                f"person {conf:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

        for face in face_infos:
            x1, y1, x2, y2 = face["bbox"]
            color = (0, 200, 0) if face["known"] else (0, 0, 255)
            text = face["name"] if face["known"] else "Unknown"
            if face["known"]:
                text = f"{text} {face['score']:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                out,
                text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        for x, y, w, h in motion.boxes:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 100), 1)

        if alarm_active:
            cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), (0, 0, 255), 8)
            cv2.putText(
                out,
                "ALARM",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

        hud_line_1 = (
            f"FPS:{fps:.1f}  Motion:{motion.pixel_count} "
            f"Faces:{len(face_infos)}  Persons:{sum(1 for d in detections if d['is_person'])}"
        )
        cv2.rectangle(out, (0, out.shape[0] - 65), (out.shape[1], out.shape[0]), (20, 20, 20), -1)
        cv2.putText(
            out,
            hud_line_1,
            (15, out.shape[0] - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            (15, out.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        return out

    def add_person_from_latest_face(self, name: str) -> Optional[dict[str, Any]]:
        with self._latest_faces_lock:
            if not self._latest_face_crops:
                return None
            selected = max(
                self._latest_face_crops,
                key=lambda img: int(img.shape[0]) * int(img.shape[1]),
            ).copy()

        result = self.registry.add_person(name=name, face_bgr=selected)
        if result:
            self.event_logger.log("REGISTRY", "Added person from live face", result)
        return result

    def add_person_from_image(self, name: str, image_bgr: np.ndarray) -> Optional[dict[str, Any]]:
        result = self.registry.add_person(name=name, face_bgr=image_bgr)
        if result:
            self.event_logger.log("REGISTRY", "Added person from image", result)
        return result

    def remove_person(self, person_id: str) -> bool:
        ok = self.registry.remove_person(person_id)
        if ok:
            self.event_logger.log("REGISTRY", "Removed person", {"person_id": person_id})
        return ok

    def list_people(self) -> list[dict[str, Any]]:
        return self.registry.list_people()

    def run(
        self,
        frame_callback: Optional[Callable[[np.ndarray, dict[str, Any]], None]] = None,
        event_callback: Optional[Callable[[dict[str, Any]], None]] = None,
        display: bool = False,
        max_frames: int = 0,
    ) -> None:
        self.reset_stop_flag()
        self._last_event_ts.clear()
        self._last_infer_detections = []
        self._latest_face_crops = []
        self._unknown_saved_descriptors = []
        self._unknown_cache_loaded = False

        cap = self._open_capture()
        previous_gray: Optional[np.ndarray] = None
        frame_idx = 0
        fps_history: list[float] = []

        self.event_logger.log("INFO", "Surveillance started", {"source": self.settings.source})

        try:
            while not self._stop_requested:
                loop_started = time.time()
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    if self._source_is_file and self.settings.loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                frame_idx += 1
                motion = self._detect_motion(previous_gray, frame_bgr)
                previous_gray = motion.current_gray

                if frame_idx % max(1, int(self.settings.infer_every_n_frames)) == 0:
                    self._last_infer_detections = self._run_inference(frame_bgr)
                detections = self._last_infer_detections

                face_infos, known_faces, unknown_faces = self._analyze_faces(frame_bgr, detections)
                person_count = sum(1 for d in detections if d["is_person"])

                motion_trigger = motion.detected
                unknown_trigger = unknown_faces > 0
                alarm_active = False

                if self.settings.enable_alarm:
                    alarm_active = (
                        (self.settings.alarm_on_motion and motion_trigger)
                        or (self.settings.alarm_on_unknown_face and unknown_trigger)
                    )

                elapsed = time.time() - loop_started
                fps = 1.0 / elapsed if elapsed > 1e-6 else 0.0
                fps_history.append(fps)
                if len(fps_history) > 30:
                    fps_history.pop(0)
                avg_fps = sum(fps_history) / len(fps_history)

                annotated = self._draw_annotations(
                    frame_bgr=frame_bgr,
                    detections=detections,
                    face_infos=face_infos,
                    motion=motion,
                    alarm_active=alarm_active,
                    fps=avg_fps,
                )

                now_ts = time.time()
                if motion_trigger and self._cooldown_ready("motion", 1.0, now_ts):
                    event = {
                        "type": "MOTION",
                        "message": "Motion detected",
                        "payload": {"pixels": motion.pixel_count, "boxes": len(motion.boxes)},
                    }
                    self.event_logger.log(event["type"], event["message"], event["payload"])
                    if event_callback:
                        event_callback(event)

                if unknown_trigger and self._cooldown_ready("unknown", 1.0, now_ts):
                    event = {
                        "type": "UNKNOWN_FACE",
                        "message": "Unknown face detected",
                        "payload": {"count": unknown_faces},
                    }
                    self.event_logger.log(event["type"], event["message"], event["payload"])
                    if event_callback:
                        event_callback(event)

                if alarm_active and self._cooldown_ready(
                    "alarm",
                    float(self.settings.alarm_cooldown_seconds),
                    now_ts,
                ):
                    sys.stdout.write("\a")
                    sys.stdout.flush()
                    event = {
                        "type": "ALARM",
                        "message": "Alarm triggered",
                        "payload": {
                            "motion_trigger": motion_trigger,
                            "unknown_trigger": unknown_trigger,
                        },
                    }
                    self.event_logger.log(event["type"], event["message"], event["payload"])
                    if event_callback:
                        event_callback(event)

                save_event_frame = (
                    self.settings.save_events
                    and (motion_trigger or unknown_trigger or alarm_active)
                    and self._cooldown_ready(
                        "snapshot",
                        float(self.settings.snapshot_cooldown_seconds),
                        now_ts,
                    )
                )
                if save_event_frame:
                    saved = self._save_frame(annotated, self.settings.events_dir, "event")
                    event = {
                        "type": "SNAPSHOT",
                        "message": "Saved event snapshot",
                        "payload": {"path": saved},
                    }
                    self.event_logger.log(event["type"], event["message"], event["payload"])
                    if event_callback:
                        event_callback(event)

                if self.settings.save_unknown_faces and unknown_trigger and self._cooldown_ready(
                    "unknown_face_save",
                    float(self.settings.snapshot_cooldown_seconds),
                    now_ts,
                ):
                    for face in face_infos:
                        if face["known"]:
                            continue
                        crop = self._safe_crop(frame_bgr, face["bbox"])
                        if crop is None:
                            continue
                        if not self._should_save_unknown_face(crop, similarity_threshold=0.97):
                            self.event_logger.log(
                                "UNKNOWN_FACE_SKIPPED",
                                "Skipped saving unknown face due to high similarity",
                                {"threshold": 0.97},
                            )
                            continue
                        saved = self._save_frame(crop, self.settings.unknown_face_dir, "unknown_face")
                        self.event_logger.log(
                            "UNKNOWN_FACE_SAVED",
                            "Unknown face image saved",
                            {"path": saved},
                        )
                        break

                stats = {
                    "frame_index": frame_idx,
                    "fps": round(avg_fps, 2),
                    "person_count": person_count,
                    "face_count": len(face_infos),
                    "known_faces": known_faces,
                    "unknown_faces": unknown_faces,
                    "motion_detected": motion_trigger,
                    "motion_pixels": motion.pixel_count,
                    "alarm_active": alarm_active,
                }

                if frame_callback:
                    frame_callback(annotated, stats)

                if display:
                    cv2.imshow(self.settings.window_title, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

                if max_frames > 0 and frame_idx >= max_frames:
                    break
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            self.event_logger.log("INFO", "Surveillance stopped", {"frame_count": frame_idx})
