import tensorflow as tf
import os

# Disable GPU to avoid CUDA configuration errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def train_cnn_model():
    print("TensorFlow Version:", tf.__version__)

    # 1. Load and Prepare Data
    print("Loading MNIST data...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape for CNN: (batch, height, width, channels)
    # MNIST is grayscale, so channels=1
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")

    # 2. Build the CNN Model
    model = tf.keras.models.Sequential([
        # Convolutional Layer 1: 32 filters, 3x3 kernel, ReLU activation
        # Input shape must be specified for the first layer
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        # Max Pooling to reduce spatial dimensions (28x28 -> 14x14)
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Convolutional Layer 2: 64 filters, 3x3 kernel
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # Max Pooling (12x12 -> 6x6 approx)
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Convolutional Layer 3: 64 filters
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten the 3D output to 1D for the dense layers
        tf.keras.layers.Flatten(),
        
        # Dense Layer: 64 neurons
        tf.keras.layers.Dense(64, activation='relu'),
        
        # Output Layer: 10 neurons (digits 0-9)
        tf.keras.layers.Dense(10)
    ])

    # 3. Compile the Model
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    # 4. Train the Model
    print("\nStarting training...")
    history = model.fit(x_train, y_train, epochs=5, validation_split=0.1)

    # 5. Evaluate the Model
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

    print(f'\nTest accuracy: {test_acc*100:.2f}%')

    # 6. Save the Model
    save_path = os.path.join(os.path.dirname(__file__), 'mnist_cnn_model.keras')
    model.save(save_path)
    print(f"CNN Model saved to {save_path}")

if __name__ == "__main__":
    train_cnn_model()
