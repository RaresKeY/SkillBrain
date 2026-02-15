import torch
from torchvision import datasets
import matplotlib.pyplot as plt
import random
import os

def observe_random_subset(num_samples=10):
    """
    Loads the MNIST dataset, selects a random subset, and saves a visualization.
    """
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    output_file = os.path.join(script_dir, 'mnist_sample.png')

    # check if data exists
    if not os.path.exists(data_dir):
        print(f"Data directory not found at {data_dir}. Please run download_dataset.py first.")
        return

    # Load the dataset (train=True for training set)
    # We don't need transforms for basic visualization, but ToTensor is useful if we wanted tensors.
    # Here we can just use the PIL images returned by default if transform is None.
    try:
        mnist_data = datasets.MNIST(root=data_dir, train=True, download=False)
    except RuntimeError:
        print("Dataset not found. Please run download_dataset.py first.")
        return

    print(f"Dataset loaded. Total images: {len(mnist_data)}")

    # Select random indices
    indices = random.sample(range(len(mnist_data)), num_samples)
    
    # Setup the plot
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    if num_samples == 1:
        axes = [axes]

    print(f"Selecting {num_samples} random images...")

    for i, idx in enumerate(indices):
        image, label = mnist_data[idx]
        
        # Plot
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Visualization saved to: {output_file}")
    print("You can open this file to observe the handwritten digits.")

if __name__ == "__main__":
    observe_random_subset(10)
