# Handwritten Character Recognition (EMNIST + CNN)

## Overview
This project implements a **handwritten character recognition system** using a **Convolutional Neural Network (CNN)** trained on the **EMNIST dataset**. The goal is to convert handwritten characters into digital text, focusing on **English letters (A–Z) and digits (0–9)**.

The project demonstrates a complete machine learning workflow including data preprocessing, model training, evaluation, and a simple user interface for testing predictions.

---

## Motivation
Handwritten character recognition is a key component of **Optical Character Recognition (OCR)** systems, which are widely used in:
- Education (digitizing handwritten notes)
- Banking (form and check processing)
- Healthcare (medical records)
- Automated data entry systems

This project explores how **deep learning**, specifically CNNs, can be applied to solve image-based classification problems.

---

## Dataset
- **EMNIST (Extended MNIST)**
- ~800,000 grayscale images
- Image size: 28×28 pixels
- Contains handwritten letters and digits

**Note:** The current implementation uses MNIST digits (0-9) as a starting point. For full EMNIST dataset with letters, download from:
https://www.nist.gov/itl/products-and-services/emnist-dataset

---

## Technologies Used
- **Python 3.8+**
- **TensorFlow / Keras** - Deep learning framework
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization
- **scikit-learn** - Evaluation metrics
- **Pillow (PIL)** - Image processing
- **OpenCV** - Computer vision (optional)
- **Tkinter** - GUI framework
- **Google Colab** - Training environment (optional)

---

## Project Structure

```
Machine-Learning-Character-Identifier/
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── data_loader.py         # Data loading and preprocessing
├── model.py               # CNN model architecture
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── gui.py                 # GUI application
├── models/                # Saved trained models (created after training)
└── results/               # Evaluation results and plots (created after evaluation)
```

---

## Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd Machine-Learning-Character-Identifier
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Data Preparation
The `data_loader.py` module handles dataset loading and preprocessing. Currently, it uses MNIST digits as a placeholder. To use the full EMNIST dataset:

1. Download EMNIST from: https://www.nist.gov/itl/products-and-services/emnist-dataset
2. Update `data_loader.py` to load the EMNIST dataset from your local files

### 2. Training the Model
Train the CNN model using the training script:

```bash
python train.py
```

This will:
- Load and preprocess the dataset
- Create a CNN model
- Train the model with validation
- Save the best model to `models/best_model.h5`
- Generate training history plots in `results/`

**Training Parameters:**
- Epochs: 50 (with early stopping)
- Batch size: 128
- Learning rate: 0.001 (with adaptive reduction)
- Validation split: 10% of training data

### 3. Evaluating the Model
Evaluate the trained model on the test set:

```bash
python evaluate.py
```

This will:
- Load the trained model
- Evaluate on test data
- Generate confusion matrix
- Show sample predictions
- Print classification report

### 4. Using the GUI
Run the graphical interface to test the model interactively:

```bash
python gui.py
```

**GUI Features:**
- Draw characters on the canvas
- Click "Predict" to get model predictions
- View confidence scores
- Clear canvas to draw new characters

---

## Model Architecture

The CNN model consists of:

1. **Convolutional Layers:**
   - Conv2D(32 filters, 3×3) → MaxPooling → Dropout(0.25)
   - Conv2D(64 filters, 3×3) → MaxPooling → Dropout(0.25)
   - Conv2D(128 filters, 3×3) → Dropout(0.25)

2. **Dense Layers:**
   - Flatten → Dense(512) → Dropout(0.5) → Dense(num_classes)

3. **Regularization:**
   - Dropout layers to prevent overfitting
   - Batch normalization (in improved model)

---

## Project Phases

### Phase 1: Data Preparation ✅
- Load EMNIST dataset
- Normalize images
- Split into train/test sets
- Evaluate different ML methods

### Phase 2: Model Development ✅
- Build baseline neural network
- Train on training set
- Implement CNN architecture

### Phase 3: Model Improvement ✅
- Tune hyperparameters
- Add regularization techniques
- Optimize with validation set

### Phase 4: Evaluation & Presentation ✅
- Evaluate on test set
- Generate visualizations
- Create GUI for demonstration

---

## Results

After training and evaluation, you'll find:
- **Training history plots** (`results/training_history.png`)
- **Confusion matrix** (`results/confusion_matrix.png`)
- **Sample predictions** (`results/sample_predictions.png`)
- **Saved models** (`models/best_model.h5`, `models/final_model.h5`)

---

## Team Members
- David Estrada
- Kohki Kita
- Burak Ozhan

---

## Future Improvements

- [ ] Implement full EMNIST dataset loading (letters + digits)
- [ ] Add data augmentation techniques
- [ ] Experiment with different architectures (ResNet, VGG)
- [ ] Support for multiple languages
- [ ] Real-time webcam input
- [ ] Mobile app integration

---

## License

This project is part of an undergraduate course assignment.

---

## References

- EMNIST Dataset: https://www.nist.gov/itl/products-and-services/emnist-dataset
- TensorFlow Documentation: https://www.tensorflow.org/
- Keras Documentation: https://keras.io/

