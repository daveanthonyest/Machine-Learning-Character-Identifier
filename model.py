"""
CNN Model Architecture for Handwritten Character Recognition
Implements a Convolutional Neural Network for classifying handwritten characters
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def create_cnn_model(input_shape=(28, 28, 1), num_classes=36):
    """
    Create a CNN model for handwritten character recognition
    
    Architecture:
    - Convolutional layers with ReLU activation
    - Max pooling layers for downsampling
    - Dropout layers for regularization
    - Dense layers for classification
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of classes to classify (36 for A-Z + 0-9)
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the model with optimizer, loss function, and metrics
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_improved_model(input_shape=(28, 28, 1), num_classes=36):
    """
    Create an improved CNN model with batch normalization
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of classes to classify
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # First convolutional block with batch normalization
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model = compile_model(model)
    return model


def print_model_summary(model):
    """Print model architecture summary"""
    model.summary()


if __name__ == "__main__":
    # Test model creation
    print("Creating CNN model...")
    model = create_cnn_model(input_shape=(28, 28, 1), num_classes=36)
    model = compile_model(model)
    
    print("\nModel Summary:")
    print_model_summary(model)

