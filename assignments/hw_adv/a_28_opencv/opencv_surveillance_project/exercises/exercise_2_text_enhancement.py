"""Exercise 2: Ancient text enhancement with robust thresholding."""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to text image")
    return parser.parse_args()


def calculate_readability(binary_image: np.ndarray) -> float:
    white_pixels = int(np.sum(binary_image == 255))
    black_pixels = int(np.sum(binary_image == 0))
    if white_pixels == 0 or black_pixels == 0:
        return 0.0
    ratio = min(white_pixels, black_pixels) / max(white_pixels, black_pixels)
    return float(ratio * 100)


def enhance_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.medianBlur(enhanced, 3)
    return enhanced


def remove_small_components(binary_inv: np.ndarray, min_area: int = 18) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_inv, connectivity=8)
    cleaned = np.zeros_like(binary_inv)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 255

    return cleaned


def resize_for_processing(image_bgr: np.ndarray, max_side: int = 1600) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image_bgr

    scale = max_side / float(longest)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)


def main() -> None:
    args = parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")

    image = resize_for_processing(image, max_side=1600)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = enhance_grayscale(image)

    _, global_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    adaptive_gaussian = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
    )
    adaptive_mean = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 11
    )

    adaptive_inv = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 11
    )
    adaptive_inv = remove_small_components(adaptive_inv, min_area=10)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    adaptive_inv = cv2.morphologyEx(adaptive_inv, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.bitwise_not(adaptive_inv)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes[0, 0].imshow(gray, cmap="gray")
    axes[0, 0].set_title("Original Grayscale")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(enhanced, cmap="gray")
    axes[0, 1].set_title("Enhanced (CLAHE + Denoise)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(global_otsu, cmap="gray")
    axes[0, 2].set_title("Global Otsu")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(adaptive_gaussian, cmap="gray")
    axes[1, 0].set_title("Adaptive Gaussian")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(adaptive_mean, cmap="gray")
    axes[1, 1].set_title("Adaptive Mean")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(cleaned, cmap="gray")
    axes[1, 2].set_title("Adaptive Cleaned")
    axes[1, 2].axis("off")

    output_dir = Path(__file__).resolve().parents[1] / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "text_enhancement_comparison.png"

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved result image: {out_path}")
    plt.show()

    print("Readability scores:")
    print(f"Global Otsu: {calculate_readability(global_otsu):.1f}/100")
    print(f"Adaptive Gaussian: {calculate_readability(adaptive_gaussian):.1f}/100")
    print(f"Adaptive Mean: {calculate_readability(adaptive_mean):.1f}/100")
    print(f"Adaptive Cleaned: {calculate_readability(cleaned):.1f}/100")


if __name__ == "__main__":
    main()
