"""CLI entrypoint for the A29 surveillance system."""

from __future__ import annotations

import argparse
from pathlib import Path

from surveillance_core import (
    APP_DIR,
    DEFAULT_SETTINGS_PATH,
    EventLogger,
    PeopleRegistry,
    SettingsStore,
    SurveillanceEngine,
    SurveillanceSettings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A29 surveillance system (CLI)")
    parser.add_argument(
        "--settings",
        type=str,
        default=str(DEFAULT_SETTINGS_PATH),
        help="Path to settings JSON file.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Input source: camera index ('0'), camera:1, local video path, or stream URL.",
    )
    parser.add_argument("--model", type=str, default=None, help="Override model path.")
    parser.add_argument("--confidence", type=float, default=None, help="Override YOLO confidence.")
    parser.add_argument("--iou", type=float, default=None, help="Override YOLO IoU.")
    parser.add_argument("--imgsz", type=int, default=None, help="Override YOLO inference image size.")
    parser.add_argument("--infer-every", type=int, default=None, help="Run inference every N frames.")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV preview window.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = full run).")
    parser.add_argument("--save-settings", action="store_true", help="Persist overrides to settings file.")
    parser.add_argument("--loop-video", action="store_true", help="Loop video file sources.")
    return parser.parse_args()


def apply_overrides(settings: SurveillanceSettings, args: argparse.Namespace) -> SurveillanceSettings:
    if args.source is not None:
        settings.source = args.source
    if args.model is not None:
        settings.model_path = args.model
    if args.confidence is not None:
        settings.confidence = float(args.confidence)
    if args.iou is not None:
        settings.iou = float(args.iou)
    if args.imgsz is not None:
        settings.imgsz = int(args.imgsz)
    if args.infer_every is not None:
        settings.infer_every_n_frames = max(1, int(args.infer_every))
    if args.loop_video:
        settings.loop_video = True
    return settings


def main() -> None:
    args = parse_args()
    settings_path = Path(args.settings)

    store = SettingsStore(settings_path)
    settings = apply_overrides(store.load(), args)
    if args.save_settings:
        store.save(settings)

    registry = PeopleRegistry(
        registry_path=APP_DIR / settings.registry_path,
        known_people_dir=APP_DIR / settings.known_people_dir,
    )
    logger = EventLogger(logs_dir=APP_DIR / settings.logs_dir)
    engine = SurveillanceEngine(settings=settings, registry=registry, event_logger=logger)

    def on_event(event: dict) -> None:
        event_type = event.get("type", "EVENT")
        message = event.get("message", "")
        payload = event.get("payload", {})
        print(f"[{event_type}] {message} {payload}")

    print("Starting CLI surveillance. Press 'q' in the OpenCV window to stop.")
    print(f"Settings file: {settings_path}")
    print(f"Source: {settings.source}")
    print(f"Model: {settings.model_path}")

    engine.run(
        frame_callback=None,
        event_callback=on_event,
        display=not args.no_display,
        max_frames=max(0, int(args.max_frames)),
    )
    print("Surveillance finished.")


if __name__ == "__main__":
    main()
