import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def compute_saliency_map(model, image, class_index):
    """
    Computes the saliency map for a given image and class.
    """
    # Convert image to tensor if not already
    image_tensor = tf.convert_to_tensor(image)
    
    # Needs a batch dimension
    if len(image_tensor.shape) == 2:
        image_tensor = tf.expand_dims(image_tensor, 0)
    if len(image_tensor.shape) == 3 and image_tensor.shape[-1] != 1: 
        # Dense input might be (1, 28, 28)
        pass 
    elif len(image_tensor.shape) == 3 and image_tensor.shape[-1] == 1:
        # CNN input might be (28, 28, 1) -> needs (1, 28, 28, 1)
        image_tensor = tf.expand_dims(image_tensor, 0)

    # Watch the input image
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        predictions = model(image_tensor)
        loss = predictions[0, class_index]

    # Get the gradients of the loss w.r.t the input image
    gradients = tape.gradient(loss, image_tensor)
    
    # Take absolute values of gradients
    gradients = tf.abs(gradients)
    
    # Squeeze to remove batch dims for visualization
    saliency = np.max(gradients, axis=-1) if len(gradients.shape) == 4 else gradients
    saliency = np.squeeze(saliency)
    
    return saliency

def visualize_focus(model_path):
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Load MNIST data
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    
    # Select random image
    idx = random.randint(0, len(x_train))
    image = x_train[idx]
    label = y_train[idx]
    
    # Preprocess for model
    # Check input shape
    input_shape = model.input_shape
    if len(input_shape) == 4:
        # CNN (28, 28, 1)
        model_input = image.reshape(1, 28, 28, 1)
    else:
        # Dense (28, 28)
        model_input = image.reshape(1, 28, 28)

    # Predict
    preds = model.predict(model_input)
    predicted_class = np.argmax(preds)
    
    print(f"True Label: {label}, Predicted: {predicted_class}")
    
    # Compute Saliency
    saliency = compute_saliency_map(model, model_input, predicted_class)
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.imshow(image, cmap='gray')
    ax1.set_title(f"Original Image (Label: {label})")
    ax1.axis('off')
    
    # Overlay heatmap
    ax2.imshow(image, cmap='gray', alpha=0.5)
    im = ax2.imshow(saliency, cmap='jet', alpha=0.7)
    ax2.set_title(f"Saliency Map (Focus for digit {predicted_class})")
    ax2.axis('off')
    
    plt.colorbar(im, ax=ax2)
    plt.tight_layout()
    
    output_file = os.path.join(os.path.dirname(__file__), 'focus_visualization.png')
    plt.savefig(output_file)
    print(f"Focus visualization saved to: {output_file}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Prefer CNN model if available as it often gives cleaner maps, but Dense works too
    cnn_path = os.path.join(script_dir, 'mnist_cnn_model.keras')
    dense_path = os.path.join(script_dir, 'mnist_model.keras')
    
    model_path = cnn_path if os.path.exists(cnn_path) else dense_path
    visualize_focus(model_path)
