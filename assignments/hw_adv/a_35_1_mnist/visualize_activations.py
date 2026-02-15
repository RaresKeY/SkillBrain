import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def visualize_dense_weights(model_path, num_neurons=16):
    """
    Visualizes the weights of the first Dense layer as images.
    """
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Find the first Dense layer that is connected to the Flatten layer
    dense_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            dense_layer = layer
            break
    
    if dense_layer is None:
        print("No Dense layer found in the model.")
        return

    # Weights shape: (Input_Features, Neurons) -> (784, 128) typically
    weights, biases = dense_layer.get_weights()
    print(f"Dense Layer Weights Shape: {weights.shape}")
    
    # Check if input is flattened image (784)
    if weights.shape[0] != 784:
        print("First dense layer does not appear to be connected directly to 28x28 input (784 weights).")
        print("This visualization is most effective for the first hidden layer.")
        return

    # Plot
    rows = int(np.sqrt(num_neurons))
    cols = int(np.ceil(num_neurons / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    fig.suptitle(f"Weights of first {num_neurons} neurons in Dense Layer 1\n(Red=Positive, Blue=Negative)", fontsize=16)
    
    for i in range(num_neurons):
        ax = axes.flatten()[i]
        if i < weights.shape[1]:
            # Extract weights for neuron i and reshape to 28x28
            neuron_weights = weights[:, i].reshape(28, 28)
            
            # Use a diverging colormap centered at 0
            im = ax.imshow(neuron_weights, cmap='bwr', vmin=-np.max(np.abs(neuron_weights)), vmax=np.max(np.abs(neuron_weights)))
            ax.axis('off')
            ax.set_title(f"Neuron {i}")
        else:
            ax.axis('off')

    plt.tight_layout()
    output_file = os.path.join(os.path.dirname(__file__), 'neuron_weights_visualization.png')
    plt.savefig(output_file)
    print(f"Visualization saved to: {output_file}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'mnist_model.keras')
    visualize_dense_weights(model_path)
