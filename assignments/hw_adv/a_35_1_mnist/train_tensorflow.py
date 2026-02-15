import tensorflow as tf
import os

# Disable GPU to avoid CUDA configuration errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def train_mnist_model():
    print("TensorFlow Version:", tf.__version__)

    # 1. Load and Prepare Data
    # We use the built-in loader which handles the IDX format automatically.
    # It caches the data to ~/.keras/datasets/mnist.npz by default.
    print("Loading MNIST data...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")

    # 2. Build the Model
    # A simple but effective architecture for MNIST
    model = tf.keras.models.Sequential([
        # Flatten 28x28 images to a 784 vector
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        
        # Hidden layer 1
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        
        # Hidden layer 2
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        
        # Output layer
        tf.keras.layers.Dense(10)
    ])

    # 3. Compile the Model
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    # 4. Train the Model
    print("\nStarting training...")
    # fit() trains the model. validation_split reserves a portion of training data to monitor progress.
    history = model.fit(x_train, y_train, epochs=5, validation_split=0.1)

    # 5. Evaluate the Model
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

    print(f'\nTest accuracy: {test_acc*100:.2f}%')

    # 6. Save the Model
    # Save in the Keras format
    save_path = os.path.join(os.path.dirname(__file__), 'mnist_model.keras')
    model.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_mnist_model()
