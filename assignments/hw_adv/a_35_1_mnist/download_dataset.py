import os
from torchvision import datasets

def download_mnist():
    """
    Downloads the MNIST dataset using torchvision.
    The dataset will be saved in the 'data' directory relative to this script.
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    print(f"Downloading MNIST dataset to: {data_dir}")
    
    # Download training data
    datasets.MNIST(root=data_dir, train=True, download=True)
    
    # Download test data
    datasets.MNIST(root=data_dir, train=False, download=True)
    
    print("Download complete.")

if __name__ == "__main__":
    download_mnist()
