"""
Training Script for Handwritten Character Recognition Model
Handles model training, hyperparameter tuning, and model saving
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

from model import create_cnn_model, compile_model, create_improved_model
from data_loader import load_emnist_letters, preprocess_data


def train_model(model, x_train, y_train, x_val, y_val, 
                epochs=50, batch_size=128, model_save_path='models'):
    """
    Train the CNN model with callbacks for best model saving and early stopping
    
    Args:
        model: Compiled Keras model
        x_train: Training images
        y_train: Training labels (categorical)
        x_val: Validation images
        y_val: Validation labels (categorical)
        epochs: Number of training epochs
        batch_size: Batch size for training
        model_save_path: Directory to save the model
    
    Returns:
        Training history
    """
    # Create models directory if it doesn't exist
    os.makedirs(model_save_path, exist_ok=True)
    
    # Define callbacks
    callbacks = [
        # Save best model based on validation accuracy
        ModelCheckpoint(
            filepath=os.path.join(model_save_path, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train the model
    print("Starting training...")
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join(model_save_path, 'final_model.h5')
    model.save(final_model_path)
    print(f"\nModel saved to {final_model_path}")
    
    return history


def plot_training_history(history, save_path='results'):
    """
    Plot training history (loss and accuracy curves)
    
    Args:
        history: Training history object from model.fit()
        save_path: Directory to save plots
    """
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_path, 'training_history.png')
    plt.savefig(plot_path)
    print(f"Training history plot saved to {plot_path}")
    plt.show()


def main():
    """Main training function"""
    print("=" * 50)
    print("Handwritten Character Recognition - Training")
    print("=" * 50)
    
    # Load and preprocess data
    print("\n1. Loading dataset...")
    (x_train, y_train), (x_test, y_test) = load_emnist_letters()
    
    print("\n2. Preprocessing data...")
    # Split training data into train and validation
    split_idx = int(0.9 * len(x_train))
    x_train_split = x_train[:split_idx]
    y_train_split = y_train[:split_idx]
    x_val = x_train[split_idx:]
    y_val = y_train[split_idx:]
    
    # Preprocess data
    num_classes = 10  # For MNIST digits (0-9)
    # For full EMNIST with letters, use num_classes=36 (A-Z + 0-9)
    
    (x_train_processed, y_train_processed), _ = preprocess_data(
        x_train_split, y_train_split, x_val, y_val, num_classes=num_classes
    )
    (_, _), (x_val_processed, y_val_processed) = preprocess_data(
        x_train_split, y_train_split, x_val, y_val, num_classes=num_classes
    )
    
    print(f"Training set: {x_train_processed.shape}")
    print(f"Validation set: {x_val_processed.shape}")
    
    # Create model
    print("\n3. Creating model...")
    model = create_cnn_model(
        input_shape=x_train_processed.shape[1:],
        num_classes=num_classes
    )
    model = compile_model(model, learning_rate=0.001)
    
    print("\nModel Architecture:")
    model.summary()
    
    # Train model
    print("\n4. Training model...")
    history = train_model(
        model,
        x_train_processed, y_train_processed,
        x_val_processed, y_val_processed,
        epochs=50,
        batch_size=128
    )
    
    # Plot training history
    print("\n5. Plotting training history...")
    plot_training_history(history)
    
    # Evaluate on test set
    print("\n6. Evaluating on test set...")
    (_, _), (x_test_processed, y_test_processed) = preprocess_data(
        x_train, y_train, x_test, y_test, num_classes=num_classes
    )
    
    test_loss, test_accuracy = model.evaluate(x_test_processed, y_test_processed, verbose=1)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()

