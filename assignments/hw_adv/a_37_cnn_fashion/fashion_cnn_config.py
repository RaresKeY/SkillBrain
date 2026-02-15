"""Fine-tuning configuration for Fashion-MNIST CNN training."""

# -----------------------------------------------------------------------------
# Reproducibility and dataset options
# -----------------------------------------------------------------------------

# Seed for NumPy/TensorFlow reproducibility.
RANDOM_SEED = 42

# Fraction of training split reserved for validation during model.fit.
VALIDATION_SPLIT = 0.1

# -----------------------------------------------------------------------------
# Hardware acceleration settings
# -----------------------------------------------------------------------------

# If True, use the first detected GPU for training when available.
USE_GPU = True

# If True, enable GPU memory growth to avoid pre-allocating all VRAM.
GPU_MEMORY_GROWTH = True

# If True and GPU is available, enable mixed precision for faster training on
# modern NVIDIA Tensor Core GPUs.
USE_MIXED_PRECISION = False

# If True, print detailed GPU debug logs during startup.
DEBUG_GPU_LOGS = True


# -----------------------------------------------------------------------------
# Input and class settings
# -----------------------------------------------------------------------------

# Fashion-MNIST image shape: 28x28 grayscale.
INPUT_SHAPE = (28, 28, 1)

# Number of target classes in Fashion-MNIST.
NUM_CLASSES = 10


# -----------------------------------------------------------------------------
# CNN architecture settings
# -----------------------------------------------------------------------------

# Convolution filter counts by block.
CONV_FILTERS = [32, 64, 128]

# Kernel size for all Conv2D layers.
KERNEL_SIZE = (3, 3)

# Hidden dense layer size after feature extraction.
DENSE_UNITS = 128

# Activation function for hidden layers.
ACTIVATION = "relu"

# Dropout rates used in feature extractor and classifier head.
DROPOUT_FEATURES = 0.25
DROPOUT_HEAD = 0.5


# -----------------------------------------------------------------------------
# Optimization and training settings
# -----------------------------------------------------------------------------

# Number of epochs (full passes over the training data).
EPOCHS = 15

# Batch size. Smaller values show more progress-bar updates per epoch.
BATCH_SIZE = 64

# Optimizer type: "adam", "sgd", or "rmsprop".
OPTIMIZER = "adam"

# Learning rate for selected optimizer.
LEARNING_RATE = 0.001

# Loss for integer labels in multi-class classification.
LOSS = "sparse_categorical_crossentropy"

# Metrics shown during training/evaluation.
METRICS = ["accuracy"]

# Keras verbosity: 0 silent, 1 progress bar, 2 one line per epoch.
VERBOSE = 1


# -----------------------------------------------------------------------------
# Regularization and callback settings
# -----------------------------------------------------------------------------

# Enable early stopping based on validation loss.
USE_EARLY_STOPPING = True

# Epochs to wait before stopping when val_loss does not improve.
EARLY_STOPPING_PATIENCE = 4

# Restore best validation-loss weights when early stopping triggers.
RESTORE_BEST_WEIGHTS = True


# -----------------------------------------------------------------------------
# Output and evaluation artifact settings
# -----------------------------------------------------------------------------

# Path where the trained Fashion CNN model will be saved.
MODEL_PATH = "fashion_cnn_model.keras"

# Directory where evaluation visuals are written.
VISUALS_DIR = "fashion_cnn_visuals"

# Number of misclassified images to show in the visual grid.
NUM_MISCLASSIFIED_TO_SHOW = 12

# Number of random/assorted test images to include in the assorted sample grid.
NUM_ASSORTED_TEST_SAMPLES = 16

# Seed used for reproducible random sampling in test visuals.
VISUALS_RANDOM_SEED = 42
