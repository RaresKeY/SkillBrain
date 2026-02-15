"""Real-time surveillance system with motion and smoke detection.

Usage:
    python surveillance_system.py
    python surveillance_system.py --camera 0 --show-debug
    python surveillance_system.py --video videos/test_video.mp4 --show-debug
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Config:
    camera_index: int = 0
    video_path: str | None = None
    motion_threshold: int = 2000
    min_contour_area: int = 1000
    smoke_pixel_threshold: int = 5000
    smoke_growth_factor: float = 1.35
    smoke_history_frames: int = 12
    frame_width: int = 960
    frame_height: int = 540
    show_debug: bool = False
    no_display: bool = False
    max_frames: int = 0


class SurveillanceSystem:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.output_dir = Path("test_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.output_dir / "event_log.txt"
        self.smoke_history: deque[int] = deque(maxlen=self.config.smoke_history_frames)

        self._last_motion_snapshot = ""
        self._last_smoke_snapshot = ""

    def run(self) -> None:
        if self.config.video_path:
            cap = cv2.VideoCapture(self.config.video_path)
            source_label = f"Video file: {self.config.video_path}"
        else:
            cap = cv2.VideoCapture(self.config.camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
            source_label = f"Camera index: {self.config.camera_index}"

        if not cap.isOpened():
            raise RuntimeError(
                "Could not open input source. Check camera index or video path."
            )

        ok, previous_frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("Failed to read initial frame from webcam.")

        previous_gray = self._preprocess_gray(previous_frame)

        print("Surveillance and Safety System started")
        print(source_label)
        print(f"Motion threshold: {self.config.motion_threshold}")
        print(f"Smoke threshold: {self.config.smoke_pixel_threshold}")
        print("Press 'q' to quit")

        frame_count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            current_gray = self._preprocess_gray(frame)
            motion_detected, motion_pixels, motion_boxes, motion_mask = self._detect_motion(
                previous_gray, current_gray
            )
            smoke_detected, smoke_pixels, smoke_mask = self._detect_smoke(frame)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            alert_text = []

            for x, y, w, h in motion_boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)

            if motion_detected:
                alert_text.append("MOTION")
                if not self._recent_snapshot(self._last_motion_snapshot, timestamp):
                    self._last_motion_snapshot = self._save_snapshot(frame, "motion", timestamp)
                    self._log_event(
                        "MOTION",
                        f"pixels={motion_pixels}, boxes={len(motion_boxes)}, file={self._last_motion_snapshot}",
                    )

            if smoke_detected:
                alert_text.append("SMOKE")
                if not self._recent_snapshot(self._last_smoke_snapshot, timestamp):
                    self._last_smoke_snapshot = self._save_snapshot(frame, "smoke", timestamp)
                    self._log_event(
                        "SMOKE",
                        f"pixels={smoke_pixels}, file={self._last_smoke_snapshot}",
                    )

            if alert_text:
                border_color = (0, 0, 255) if "MOTION" in alert_text else (0, 100, 255)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), border_color, 10)
                cv2.putText(
                    frame,
                    " | ".join(alert_text) + " ALERT",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    border_color,
                    3,
                )

            cv2.putText(
                frame,
                f"Motion pixels: {motion_pixels}",
                (15, frame.shape[0] - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Smoke pixels: {smoke_pixels}",
                (15, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            if not self.config.no_display:
                cv2.imshow("Surveillance System", frame)
                if self.config.show_debug:
                    cv2.imshow("Motion Mask", motion_mask)
                    cv2.imshow("Smoke Mask", smoke_mask)

            previous_gray = current_gray
            frame_count += 1

            if self.config.max_frames > 0 and frame_count >= self.config.max_frames:
                break

            if not self.config.no_display and cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        if not self.config.no_display:
            cv2.destroyAllWindows()
        print("System stopped")

    def _preprocess_gray(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (21, 21), 0)

    def _detect_motion(
        self, previous_gray: np.ndarray, current_gray: np.ndarray
    ) -> tuple[bool, int, list[tuple[int, int, int, int]], np.ndarray]:
        frame_delta = cv2.absdiff(previous_gray, current_gray)
        _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[tuple[int, int, int, int]] = []
        total_pixels = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_contour_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))
            total_pixels += int(area)

        detected = total_pixels > self.config.motion_threshold
        return detected, total_pixels, boxes, thresh

    def _detect_smoke(self, frame: np.ndarray) -> tuple[bool, int, np.ndarray]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_smoke = np.array([0, 0, 120], dtype=np.uint8)
        upper_smoke = np.array([180, 65, 255], dtype=np.uint8)

        smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)
        kernel = np.ones((5, 5), np.uint8)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)

        smoke_pixels = int(np.sum(smoke_mask > 0))
        self.smoke_history.append(smoke_pixels)

        smoke_detected = smoke_pixels > self.config.smoke_pixel_threshold
        if len(self.smoke_history) >= 8:
            midpoint = len(self.smoke_history) // 2
            older = np.mean(list(self.smoke_history)[:midpoint])
            recent = np.mean(list(self.smoke_history)[midpoint:])
            if older > 0 and recent > older * self.config.smoke_growth_factor:
                smoke_detected = True

        return smoke_detected, smoke_pixels, smoke_mask

    def _save_snapshot(self, frame: np.ndarray, prefix: str, timestamp: str) -> str:
        file_name = f"{prefix}_{timestamp}.jpg"
        path = self.output_dir / file_name
        cv2.imwrite(str(path), frame)
        return file_name

    def _log_event(self, event_type: str, details: str) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{now}] {event_type}: {details}\n"
        print(line.strip())
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line)

    @staticmethod
    def _recent_snapshot(last_name: str, now_timestamp: str) -> bool:
        if not last_name:
            return False
        # Prevent writing many snapshots in the same second.
        return now_timestamp in last_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenCV surveillance and smoke detection system")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to a pre-recorded video file. If provided, camera is ignored.",
    )
    parser.add_argument("--show-debug", action="store_true", help="Show motion/smoke mask windows")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without GUI windows (useful for headless environments).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop automatically after N frames (0 = process full stream).",
    )
    parser.add_argument("--motion-threshold", type=int, default=2000, help="Motion alert pixel threshold")
    parser.add_argument("--min-contour-area", type=int, default=1000, help="Minimum moving contour area")
    parser.add_argument("--smoke-threshold", type=int, default=5000, help="Smoke alert pixel threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        camera_index=args.camera,
        video_path=args.video,
        motion_threshold=args.motion_threshold,
        min_contour_area=args.min_contour_area,
        smoke_pixel_threshold=args.smoke_threshold,
        show_debug=args.show_debug,
        no_display=args.no_display,
        max_frames=args.max_frames,
    )

    system = SurveillanceSystem(config)
    system.run()


if __name__ == "__main__":
    main()
