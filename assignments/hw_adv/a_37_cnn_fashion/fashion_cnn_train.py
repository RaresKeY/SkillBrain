"""Train a configurable CNN on the Fashion-MNIST dataset."""

from __future__ import annotations

import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
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


def set_reproducibility(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def log_gpu_debug_info() -> None:
    """Print detailed diagnostics to confirm TensorFlow GPU execution."""
    print("[DEBUG] TensorFlow version:", tf.__version__)
    print("[DEBUG] Built with CUDA:", tf.test.is_built_with_cuda())
    print("[DEBUG] Built with GPU support:", tf.test.is_built_with_gpu_support())

    physical_gpus = tf.config.list_physical_devices("GPU")
    logical_gpus = tf.config.list_logical_devices("GPU")
    print("[DEBUG] Physical GPUs:", physical_gpus)
    print("[DEBUG] Logical GPUs:", logical_gpus)

    a = tf.random.uniform((1024, 1024))
    b = tf.random.uniform((1024, 1024))
    c = tf.matmul(a, b)
    _ = c.numpy()
    print("[DEBUG] Default matmul tensor device:", c.device)

    if logical_gpus:
        try:
            with tf.device("/GPU:0"):
                g = tf.matmul(a, b)
                _ = g.numpy()
            print("[DEBUG] Forced /GPU:0 matmul tensor device:", g.device)
        except Exception as exc:
            print("[DEBUG] Forced /GPU:0 test failed:", exc)


def configure_hardware_acceleration() -> None:
    """Configure GPU usage and optional mixed precision."""
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus or not cfg.USE_GPU:
        print("[-] GPU acceleration disabled or no GPU found. Using CPU.")
        mixed_precision.set_global_policy("float32")
        if cfg.DEBUG_GPU_LOGS:
            log_gpu_debug_info()
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

    if cfg.DEBUG_GPU_LOGS:
        log_gpu_debug_info()


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Fashion-MNIST and normalize to [0, 1]."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return x_train, y_train, x_test, y_test


def create_optimizer() -> tf.keras.optimizers.Optimizer:
    """Create optimizer from config values."""
    name = cfg.OPTIMIZER.lower()
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE)
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=cfg.LEARNING_RATE)
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=cfg.LEARNING_RATE)
    raise ValueError(f"Unsupported optimizer: {cfg.OPTIMIZER}")


def build_model() -> tf.keras.Model:
    """Build CNN model from config."""
    model = tf.keras.Sequential(name="fashion_mnist_cnn")
    model.add(tf.keras.layers.Input(shape=cfg.INPUT_SHAPE))

    model.add(
        tf.keras.layers.Conv2D(
            cfg.CONV_FILTERS[0],
            cfg.KERNEL_SIZE,
            activation=cfg.ACTIVATION,
            padding="same",
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            cfg.CONV_FILTERS[0],
            cfg.KERNEL_SIZE,
            activation=cfg.ACTIVATION,
            padding="same",
        )
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(
        tf.keras.layers.Conv2D(
            cfg.CONV_FILTERS[1],
            cfg.KERNEL_SIZE,
            activation=cfg.ACTIVATION,
            padding="same",
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            cfg.CONV_FILTERS[1],
            cfg.KERNEL_SIZE,
            activation=cfg.ACTIVATION,
            padding="same",
        )
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(cfg.DROPOUT_FEATURES))

    model.add(
        tf.keras.layers.Conv2D(
            cfg.CONV_FILTERS[2],
            cfg.KERNEL_SIZE,
            activation=cfg.ACTIVATION,
            padding="same",
        )
    )

    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(cfg.DENSE_UNITS, activation=cfg.ACTIVATION))
    model.add(tf.keras.layers.Dropout(cfg.DROPOUT_HEAD))
    model.add(tf.keras.layers.Dense(cfg.NUM_CLASSES, activation="softmax"))

    model.compile(
        optimizer=create_optimizer(),
        loss=cfg.LOSS,
        metrics=cfg.METRICS,
    )

    return model


def train_model(model: tf.keras.Model, x_train: np.ndarray, y_train: np.ndarray) -> tf.keras.callbacks.History:
    """Train the model using config-driven settings."""
    callbacks: list[tf.keras.callbacks.Callback] = []

    if cfg.USE_EARLY_STOPPING:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg.EARLY_STOPPING_PATIENCE,
                restore_best_weights=cfg.RESTORE_BEST_WEIGHTS,
            )
        )

    return model.fit(
        x_train,
        y_train,
        validation_split=cfg.VALIDATION_SPLIT,
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        callbacks=callbacks,
        verbose=cfg.VERBOSE,
    )


def evaluate_model(model: tf.keras.Model, x_test: np.ndarray, y_test: np.ndarray) -> None:
    """Evaluate and print test metrics + classification report."""
    loss, accuracy = model.evaluate(x_test, y_test, verbose=cfg.VERBOSE)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))


def main() -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    tf.keras.utils.enable_interactive_logging()

    configure_hardware_acceleration()
    set_reproducibility(cfg.RANDOM_SEED)
    x_train, y_train, x_test, y_test = load_data()

    model = build_model()
    model.summary()

    train_model(model, x_train, y_train)
    evaluate_model(model, x_test, y_test)
    model.save(cfg.MODEL_PATH)
    print(f"\n[-] Saved model to: {cfg.MODEL_PATH}")


if __name__ == "__main__":
    main()
