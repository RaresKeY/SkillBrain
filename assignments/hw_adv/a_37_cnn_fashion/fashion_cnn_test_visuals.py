"""Evaluate trained Fashion-MNIST CNN and generate visual diagnostics."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import mixed_precision

import fashion_cnn_config as cfg


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def configure_hardware_acceleration() -> None:
    """Configure GPU usage for inference/evaluation."""
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus or not cfg.USE_GPU:
        print("[-] GPU acceleration disabled or no GPU found. Using CPU.")
        mixed_precision.set_global_policy("float32")
        return

    if cfg.GPU_MEMORY_GROWTH:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    if cfg.USE_MIXED_PRECISION:
        mixed_precision.set_global_policy("mixed_float16")
        print(f"[-] GPU detected ({len(gpus)} device(s)). Mixed precision enabled.")
    else:
        mixed_precision.set_global_policy("float32")
        print(f"[-] GPU detected ({len(gpus)} device(s)). Using float32 precision.")


def load_test_data() -> tuple[np.ndarray, np.ndarray]:
    """Load Fashion-MNIST test data and normalize to [0, 1]."""
    _, (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, axis=-1)
    return x_test, y_test


def ensure_visuals_dir() -> None:
    """Create output directory for visual artifacts."""
    os.makedirs(cfg.VISUALS_DIR, exist_ok=True)


def plot_confusion_matrix(cm: np.ndarray, out_path: str) -> None:
    """Save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_title("Fashion-MNIST Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)

    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_misclassified_examples(
    x_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    out_path: str,
) -> None:
    """Save a grid of misclassified test samples."""
    wrong_idx = np.where(y_test != y_pred)[0]
    max_items = cfg.NUM_MISCLASSIFIED_TO_SHOW

    if wrong_idx.size == 0:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.axis("off")
        ax.set_title("No misclassified samples in test set")
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return

    shown_idx = wrong_idx[:max_items]
    cols = 4
    rows = int(np.ceil(len(shown_idx) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.2 * rows))
    axes = np.array(axes).reshape(-1)

    for plot_i, sample_i in enumerate(shown_idx):
        ax = axes[plot_i]
        ax.imshow(x_test[sample_i].squeeze(), cmap="gray")
        true_name = CLASS_NAMES[y_test[sample_i]]
        pred_name = CLASS_NAMES[y_pred[sample_i]]
        conf = float(np.max(y_prob[sample_i]))
        ax.set_title(f"T: {true_name}\nP: {pred_name} ({conf:.2f})", fontsize=9)
        ax.axis("off")

    for ax in axes[len(shown_idx) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_prediction_samples(
    x_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    num_samples: int = 12,
) -> None:
    """Save a grid of test predictions (correct and incorrect)."""
    num_samples = min(num_samples, len(x_test))
    cols = 4
    rows = int(np.ceil(num_samples / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.2 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(num_samples):
        ax = axes[i]
        ax.imshow(x_test[i].squeeze(), cmap="gray")
        true_name = CLASS_NAMES[y_test[i]]
        pred_name = CLASS_NAMES[y_pred[i]]
        color = "green" if y_test[i] == y_pred[i] else "red"
        ax.set_title(f"T: {true_name}\nP: {pred_name}", color=color, fontsize=9)
        ax.axis("off")

    for ax in axes[num_samples:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_assorted_test_samples(
    x_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    out_path: str,
) -> None:
    """Save a grid of randomly assorted test images."""
    rng = np.random.default_rng(cfg.VISUALS_RANDOM_SEED)
    num_samples = min(cfg.NUM_ASSORTED_TEST_SAMPLES, len(x_test))
    sampled_idx = rng.choice(len(x_test), size=num_samples, replace=False)

    cols = 4
    rows = int(np.ceil(num_samples / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.2 * rows))
    axes = np.array(axes).reshape(-1)

    for plot_i, sample_i in enumerate(sampled_idx):
        ax = axes[plot_i]
        ax.imshow(x_test[sample_i].squeeze(), cmap="gray")
        true_name = CLASS_NAMES[y_test[sample_i]]
        pred_name = CLASS_NAMES[y_pred[sample_i]]
        conf = float(np.max(y_prob[sample_i]))
        color = "green" if y_test[sample_i] == y_pred[sample_i] else "red"
        ax.set_title(f"T: {true_name}\nP: {pred_name} ({conf:.2f})", color=color, fontsize=9)
        ax.axis("off")

    for ax in axes[num_samples:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    tf.keras.utils.enable_interactive_logging()

    configure_hardware_acceleration()
    ensure_visuals_dir()

    if not os.path.exists(cfg.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {cfg.MODEL_PATH}. "
            "Run fashion_cnn_train.py first to train and save the model."
        )

    model = tf.keras.models.load_model(cfg.MODEL_PATH)
    x_test, y_test = load_test_data()

    loss, accuracy = model.evaluate(x_test, y_test, verbose=cfg.VERBOSE)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    y_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_test, y_pred)
    cm_path = os.path.join(cfg.VISUALS_DIR, "confusion_matrix.png")
    misclf_path = os.path.join(cfg.VISUALS_DIR, "misclassified_examples.png")
    samples_path = os.path.join(cfg.VISUALS_DIR, "prediction_samples.png")
    assorted_path = os.path.join(cfg.VISUALS_DIR, "assorted_test_samples.png")

    plot_confusion_matrix(cm, cm_path)
    plot_misclassified_examples(x_test, y_test, y_pred, y_prob, misclf_path)
    plot_prediction_samples(x_test, y_test, y_pred, samples_path)
    plot_assorted_test_samples(x_test, y_test, y_pred, y_prob, assorted_path)

    print("\n[-] Saved visuals:")
    print(f"    - {cm_path}")
    print(f"    - {misclf_path}")
    print(f"    - {samples_path}")
    print(f"    - {assorted_path}")


if __name__ == "__main__":
    main()
