"""TensorFlow Iris classification training script.

This script trains a simple feed-forward neural network on the Iris dataset.
Training knobs are kept in `finetune_config.py`.
"""

from __future__ import annotations

import os

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import finetune_config as cfg


def set_reproducibility(seed: int) -> None:
    """Set seeds for reproducible behavior where possible."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_and_prepare_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load Iris dataset, split train/test, and standardize features."""
    iris = load_iris()
    x = iris.data
    y = iris.target

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_SEED,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test, iris.target_names.tolist()


def create_optimizer() -> tf.keras.optimizers.Optimizer:
    """Create optimizer based on config."""
    optimizer_name = cfg.OPTIMIZER.lower()
    if optimizer_name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE)
    if optimizer_name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=cfg.LEARNING_RATE)
    if optimizer_name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=cfg.LEARNING_RATE)
    raise ValueError(f"Unsupported optimizer: {cfg.OPTIMIZER}")


def build_model() -> tf.keras.Model:
    """Build a configurable MLP for Iris classification."""
    l2_reg = tf.keras.regularizers.L2(cfg.L2_REG_FACTOR)

    model = tf.keras.Sequential(name="iris_mlp")
    model.add(tf.keras.layers.Input(shape=(cfg.INPUT_DIM,)))

    for units in cfg.HIDDEN_UNITS:
        model.add(
            tf.keras.layers.Dense(
                units,
                activation=cfg.ACTIVATION,
                kernel_regularizer=l2_reg,
            )
        )
        if cfg.DROPOUT_RATE > 0:
            model.add(tf.keras.layers.Dropout(cfg.DROPOUT_RATE))

    model.add(tf.keras.layers.Dense(cfg.NUM_CLASSES, activation="softmax"))

    model.compile(
        optimizer=create_optimizer(),
        loss=cfg.LOSS,
        metrics=cfg.METRICS,
    )
    return model


def train_model(model: tf.keras.Model, x_train: np.ndarray, y_train: np.ndarray) -> tf.keras.callbacks.History:
    """Train the model using config-driven options."""
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


def evaluate_model(model: tf.keras.Model, x_test: np.ndarray, y_test: np.ndarray, class_names: list[str]) -> None:
    """Evaluate and print metrics and per-class report."""
    loss, accuracy = model.evaluate(x_test, y_test, verbose=cfg.VERBOSE)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    y_proba = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))


def main() -> None:
    # Ensure relative imports and run behavior are stable from any launch path.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    tf.keras.utils.enable_interactive_logging()

    set_reproducibility(cfg.RANDOM_SEED)
    x_train, x_test, y_train, y_test, class_names = load_and_prepare_data()

    model = build_model()
    model.summary()

    train_model(model, x_train, y_train)
    evaluate_model(model, x_test, y_test, class_names)


if __name__ == "__main__":
    main()
