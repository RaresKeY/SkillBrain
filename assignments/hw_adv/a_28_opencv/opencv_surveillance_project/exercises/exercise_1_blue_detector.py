"""Exercise 1: Detect blue objects using HSV and contour analysis."""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--min-area", type=int, default=1200, help="Minimum contour area to keep")
    return parser.parse_args()


def build_blue_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([95, 70, 40], dtype=np.uint8)
    upper_blue = np.array([135, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Smooth small gaps and remove tiny isolated noise.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_blue_regions(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int, float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = mask.shape
    image_area = float(h * w)
    max_area = image_area * 0.20
    border_margin = 6

    regions: list[tuple[int, int, int, int, float]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        touches_border = (
            x <= border_margin
            or y <= border_margin
            or (x + bw) >= (w - border_margin)
            or (y + bh) >= (h - border_margin)
        )
        if touches_border:
            continue

        regions.append((x, y, bw, bh, area))

    return sorted(regions, key=lambda r: r[4], reverse=True)


def visualize_and_save(image_bgr: np.ndarray, mask: np.ndarray, regions: list[tuple[int, int, int, int, float]]) -> None:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    detected = image_rgb.copy()

    for idx, (x, y, w, h, area) in enumerate(regions, start=1):
        cv2.rectangle(detected, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            detected,
            f"#{idx} A:{int(area)}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Blue Mask (Refined)")
    axes[1].axis("off")

    axes[2].imshow(detected)
    axes[2].set_title(f"Detected Blue Regions: {len(regions)}")
    axes[2].axis("off")

    output_dir = Path(__file__).resolve().parents[1] / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "blue_detection_result.png"

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved result image: {out_path}")
    plt.show()


def main() -> None:
    args = parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")

    mask = build_blue_mask(image)
    regions = detect_blue_regions(mask, args.min_area)

    print(f"Detected blue objects: {len(regions)}")
    for idx, (_, _, _, _, area) in enumerate(regions, start=1):
        print(f"  Object {idx}: area={area:.1f} px")

    visualize_and_save(image, mask, regions)


if __name__ == "__main__":
    main()
