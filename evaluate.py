"""
Evaluation Script for Handwritten Character Recognition Model
Provides detailed evaluation metrics and visualizations
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from data_loader import load_emnist_letters, preprocess_data


def load_trained_model(model_path='models/best_model.h5'):
    """
    Load a trained model from file
    
    Args:
        model_path: Path to saved model file
    
    Returns:
        Loaded Keras model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    model = keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model


def evaluate_model(model, x_test, y_test, class_names=None):
    """
    Evaluate model on test set and return metrics
    
    Args:
        model: Trained Keras model
        x_test: Test images
        y_test: Test labels (categorical)
        class_names: Optional list of class names
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    # Get predictions
    predictions = model.predict(x_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Calculate additional metrics
    correct_predictions = np.sum(predicted_classes == true_classes)
    total_samples = len(true_classes)
    
    metrics = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'correct_predictions': correct_predictions,
        'total_samples': total_samples,
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'true_classes': true_classes
    }
    
    return metrics


def plot_confusion_matrix(true_classes, predicted_classes, class_names=None, save_path='results'):
    """
    Plot and save confusion matrix
    
    Args:
        true_classes: True class labels
        predicted_classes: Predicted class labels
        class_names: Optional list of class names
        save_path: Directory to save the plot
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Generate confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names if class_names else range(len(cm)),
                yticklabels=class_names if class_names else range(len(cm)))
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_path, 'confusion_matrix.png')
    plt.savefig(plot_path)
    print(f"Confusion matrix saved to {plot_path}")
    plt.show()


def plot_sample_predictions(model, x_test, y_test, num_samples=10, class_names=None, save_path='results'):
    """
    Plot sample predictions with true and predicted labels
    
    Args:
        model: Trained Keras model
        x_test: Test images
        y_test: Test labels (categorical)
        num_samples: Number of samples to display
        class_names: Optional list of class names
        save_path: Directory to save the plot
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Get random samples
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    # Get predictions
    sample_images = x_test[indices]
    sample_labels = np.argmax(y_test[indices], axis=1)
    predictions = model.predict(sample_images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    confidence = np.max(predictions, axis=1)
    
    # Plot samples
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        img = sample_images[i]
        
        # Remove channel dimension if present
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = img.squeeze()
        
        axes[i].imshow(img, cmap='gray')
        
        true_label = sample_labels[i]
        pred_label = predicted_labels[i]
        conf = confidence[i]
        
        if class_names:
            true_text = class_names[true_label] if true_label < len(class_names) else str(true_label)
            pred_text = class_names[pred_label] if pred_label < len(class_names) else str(pred_label)
        else:
            true_text = str(true_label)
            pred_text = str(pred_label)
        
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {true_text}\nPred: {pred_text}\nConf: {conf:.2f}', 
                         color=color, fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_path, 'sample_predictions.png')
    plt.savefig(plot_path)
    print(f"Sample predictions plot saved to {plot_path}")
    plt.show()


def print_classification_report(true_classes, predicted_classes, class_names=None):
    """
    Print detailed classification report
    
    Args:
        true_classes: True class labels
        predicted_classes: Predicted class labels
        class_names: Optional list of class names
    """
    if class_names is None:
        class_names = [str(i) for i in range(max(true_classes) + 1)]
    
    report = classification_report(
        true_classes, 
        predicted_classes, 
        target_names=class_names,
        output_dict=False
    )
    
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(report)


def main():
    """Main evaluation function"""
    print("=" * 50)
    print("Handwritten Character Recognition - Evaluation")
    print("=" * 50)
    
    # Load model
    print("\n1. Loading trained model...")
    model_path = 'models/best_model.h5'
    if not os.path.exists(model_path):
        model_path = 'models/final_model.h5'
    
    try:
        model = load_trained_model(model_path)
    except FileNotFoundError:
        print("Error: No trained model found. Please run train.py first.")
        return
    
    # Load and preprocess test data
    print("\n2. Loading test data...")
    (x_train, y_train), (x_test, y_test) = load_emnist_letters()
    
    print("\n3. Preprocessing test data...")
    num_classes = 10  # For MNIST digits
    # For full EMNIST with letters, use num_classes=36
    
    (_, _), (x_test_processed, y_test_processed) = preprocess_data(
        x_train, y_train, x_test, y_test, num_classes=num_classes
    )
    
    # Evaluate model
    print("\n4. Evaluating model...")
    metrics = evaluate_model(model, x_test_processed, y_test_processed)
    
    print(f"\nTest Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test Loss: {metrics['test_loss']:.4f}")
    print(f"Correct Predictions: {metrics['correct_predictions']}/{metrics['total_samples']}")
    
    # Generate classification report
    print("\n5. Generating classification report...")
    class_names = [str(i) for i in range(num_classes)]  # For digits 0-9
    print_classification_report(
        metrics['true_classes'], 
        metrics['predicted_classes'],
        class_names=class_names
    )
    
    # Plot confusion matrix
    print("\n6. Plotting confusion matrix...")
    plot_confusion_matrix(
        metrics['true_classes'],
        metrics['predicted_classes'],
        class_names=class_names
    )
    
    # Plot sample predictions
    print("\n7. Plotting sample predictions...")
    plot_sample_predictions(
        model,
        x_test_processed,
        y_test_processed,
        num_samples=10,
        class_names=class_names
    )
    
    print("\n" + "=" * 50)
    print("Evaluation completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()

