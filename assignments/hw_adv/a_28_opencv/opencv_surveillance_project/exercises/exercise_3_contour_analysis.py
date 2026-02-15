"""Exercise 3: Contour area analysis for large objects."""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to objects image")
    parser.add_argument("--min-area", type=float, default=600.0, help="Minimum contour area")
    return parser.parse_args()


def random_color(index: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(seed=index + 101)
    return tuple(int(v) for v in rng.integers(40, 255, 3))


def build_object_mask(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (5, 5), 0)

    # Automatic Canny thresholds from image median improve portability.
    median = float(np.median(blur))
    lower = int(max(0, (1.0 - 0.33) * median))
    upper = int(min(255, (1.0 + 0.33) * median))
    edges = cv2.Canny(blur, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def analyze_contours(
    mask: np.ndarray, min_area: float
) -> tuple[list[dict[str, float | tuple[int, int]]], list[np.ndarray]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(mask.shape[0] * mask.shape[1])
    max_area = image_area * 0.30

    analyzed: list[dict[str, float | tuple[int, int]]] = []
    kept_contours: list[np.ndarray] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        analyzed.append(
            {
                "area": area,
                "perimeter": perimeter,
                "centroid": (cx, cy),
            }
        )
        kept_contours.append(contour)

    order = np.argsort([-item["area"] for item in analyzed])
    analyzed = [analyzed[i] for i in order]
    kept_contours = [kept_contours[i] for i in order]
    return analyzed, kept_contours


def main() -> None:
    args = parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask = build_object_mask(gray)
    metrics, contours = analyze_contours(mask, args.min_area)

    analyzed = image_rgb.copy()
    for idx, (contour, info) in enumerate(zip(contours, metrics), start=1):
        color = random_color(idx)
        cv2.drawContours(analyzed, [contour], -1, color, 2)

        cx, cy = info["centroid"]
        cv2.putText(
            analyzed,
            f"#{idx} A:{int(info['area'])} P:{int(info['perimeter'])}",
            (cx + 4, max(16, cy - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )

    for idx, info in enumerate(metrics, start=1):
        cx, cy = info["centroid"]
        print(
            f"Object {idx}: area={info['area']:.1f} px, perimeter={info['perimeter']:.1f} px, "
            f"centroid=({cx}, {cy})"
        )

    print(f"Total objects >{args.min_area:.0f} px: {len(metrics)}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Edge-Derived Object Mask")
    axes[1].axis("off")

    axes[2].imshow(analyzed)
    axes[2].set_title("Large Contour Analysis")
    axes[2].axis("off")

    output_dir = Path(__file__).resolve().parents[1] / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "contour_analysis_result.png"

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved result image: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
