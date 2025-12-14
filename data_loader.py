"""
Data Loading and Preprocessing Module
Handles EMNIST dataset loading, normalization, and train/test splitting
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt


def load_emnist_letters():
    """
    Load EMNIST Letters dataset (A-Z, a-z)
    Note: EMNIST is not directly available in Keras, so we'll use a workaround
    For full EMNIST, you may need to download from: https://www.nist.gov/itl/products-and-services/emnist-dataset
    
    This function provides a placeholder that can be adapted for actual EMNIST data.
    """
    try:
        # Try to load EMNIST if available
        # For now, we'll use MNIST digits as a starting point
        # In production, replace this with actual EMNIST loading code
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Normalize to 0-1 range
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape for CNN (add channel dimension)
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        
        return (x_train, y_train), (x_test, y_test)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Note: For full EMNIST dataset, download from:")
        print("https://www.nist.gov/itl/products-and-services/emnist-dataset")
        raise


def load_custom_emnist(path=None):
    """
    Load EMNIST dataset from custom path if downloaded locally
    This is a placeholder for actual EMNIST loading implementation
    """
    if path is None:
        print("No custom path provided. Using MNIST as fallback.")
        return load_emnist_letters()
    
    # TODO: Implement actual EMNIST loading from .mat or .npz files
    # EMNIST typically comes in .mat format (MATLAB) or can be converted to .npz
    pass


def preprocess_data(x_train, y_train, x_test, y_test, num_classes=36):
    """
    Preprocess data for training
    
    Args:
        x_train: Training images
        y_train: Training labels
        x_test: Test images
        y_test: Test labels
        num_classes: Number of classes (26 letters + 10 digits = 36 for A-Z, 0-9)
    
    Returns:
        Preprocessed data ready for CNN training
    """
    # Ensure data is normalized
    if x_train.max() > 1.0:
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
    
    # Ensure proper shape for CNN
    if len(x_train.shape) == 3:
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
    
    # Convert labels to categorical (one-hot encoding)
    y_train_categorical = keras.utils.to_categorical(y_train, num_classes)
    y_test_categorical = keras.utils.to_categorical(y_test, num_classes)
    
    return (x_train, y_train_categorical), (x_test, y_test_categorical)


def visualize_samples(x_data, y_data, num_samples=10, class_names=None):
    """
    Visualize sample images from the dataset
    
    Args:
        x_data: Image data
        y_data: Labels
        num_samples: Number of samples to display
        class_names: Optional list of class names for display
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        idx = np.random.randint(0, len(x_data))
        img = x_data[idx]
        
        # Remove channel dimension if present for display
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = img.squeeze()
        
        axes[i].imshow(img, cmap='gray')
        label = y_data[idx]
        if class_names:
            label_text = class_names[label] if label < len(class_names) else str(label)
        else:
            label_text = str(label)
        axes[i].set_title(f'Label: {label_text}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test data loading
    print("Loading dataset...")
    (x_train, y_train), (x_test, y_test) = load_emnist_letters()
    
    print(f"Training set shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test set shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # Visualize samples
    print("\nVisualizing sample images...")
    visualize_samples(x_train, y_train, num_samples=10)

