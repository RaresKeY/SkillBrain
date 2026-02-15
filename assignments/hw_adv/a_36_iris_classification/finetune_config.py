"""Fine-tuning configuration for TensorFlow Iris classification.

Update values in this file to run quick experiments without changing
training logic in `iris_classification.py`.
"""

# -----------------------------------------------------------------------------
# Reproducibility and data split settings
# -----------------------------------------------------------------------------

# Random seed used across NumPy and TensorFlow to make runs reproducible.
RANDOM_SEED = 42

# Fraction of full dataset used as held-out test set.
TEST_SIZE = 0.2

# Fraction of training data used for validation during `model.fit`.
VALIDATION_SPLIT = 0.2


# -----------------------------------------------------------------------------
# Model architecture settings
# -----------------------------------------------------------------------------

# Number of input features in the Iris dataset (sepal/petal measurements).
INPUT_DIM = 4

# Number of prediction classes in the Iris dataset.
NUM_CLASSES = 3

# Hidden layer sizes used in the MLP classifier.
HIDDEN_UNITS = [32, 16]

# Activation function used for hidden layers (e.g., "relu", "tanh", "elu").
ACTIVATION = "relu"

# Dropout rate for regularization; 0.2 means randomly drop 20% units at train time.
DROPOUT_RATE = 0.2


# -----------------------------------------------------------------------------
# Optimization and training settings
# -----------------------------------------------------------------------------

# Number of training epochs (full passes through training data).
EPOCHS = 100

# Number of samples processed before each gradient update.
# Use a small batch so each epoch has many steps and the Keras verbose=1 bar visibly redraws.
BATCH_SIZE = 4

# Learning rate controlling optimizer step size.
LEARNING_RATE = 0.001

# Optimizer name ("adam", "sgd", "rmsprop") used to minimize training loss.
OPTIMIZER = "adam"

# Loss function for multi-class classification with integer labels.
LOSS = "sparse_categorical_crossentropy"

# Metrics reported while training and evaluation.
METRICS = ["accuracy"]


# -----------------------------------------------------------------------------
# Callback and regularization settings
# -----------------------------------------------------------------------------

# If True, stop early when validation loss stops improving.
USE_EARLY_STOPPING = True

# Number of epochs to wait for validation-loss improvement before stopping.
EARLY_STOPPING_PATIENCE = 12

# If True, restore best validation-loss weights after early stopping.
RESTORE_BEST_WEIGHTS = True

# L2 regularization factor applied to Dense layer kernels.
L2_REG_FACTOR = 1e-4

# Print verbosity passed to Keras fit/evaluate (0=silent, 1=progress bar, 2=one line/epoch).
VERBOSE = 1
