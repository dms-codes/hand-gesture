# Hand Gesture Recognition using Deep Learning Models

This project implements multiple deep learning models to recognize hand gestures using a dataset of grayscale images. The supported architectures include CNN, LeNet-5, AlexNet, and VGGNet. Each model is evaluated for its effectiveness, accuracy, and resource efficiency, allowing users to select a model suitable for their specific application.

---

## Features
- **Supports Multiple Architectures**: CNN, LeNet-5, AlexNet, and VGGNet.
- **Model Evaluation**: Generates confusion matrices, classification reports, and accuracy/loss graphs.
- **Custom Dataset Handling**: Creates datasets from labeled gesture images.
- **Model Persistence**: Saves and loads trained models for future use.
- **Plug-and-Play**: Easily switch between different architectures with minimal changes.

---

## Prerequisites
Before running the project, ensure you have the following dependencies installed:

- Python (>= 3.7)
- NumPy
- Pillow (PIL)
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow/Keras
- pickle (standard Python library)

To install missing packages, run:

```bash
pip install numpy pillow matplotlib seaborn scikit-learn tensorflow
```

---

## File Structure
- **Input Data**: Gesture images should be stored in the `input/leapGestRecog` directory, organized into subdirectories representing gesture classes.
- **Model Files**: Saved models are stored with architecture-specific names (e.g., `gesture_model_cnn.keras`, `gesture_model_alexnet.keras`).
- **Reports and Plots**:
  - Confusion matrices (`confusion_matrix_<model>.png`)
  - Training history graphs (`training_history_<model>.png`)
  - Classification reports (`report_model_<model>.txt`)

---

## How to Use

### 1. Dataset Preparation
Place gesture images in the `input/leapGestRecog` directory. Ensure the folder structure is organized with subdirectories named after each gesture class.

### 2. Running the Program
Execute the script by running:

```bash
python main.py
```

### 3. Model Selection
To switch between models, update the `MODEL_TYPE` variable in the `__main__` block:
```python
MODEL_TYPE = MODEL_TYPE_CNN       # For CNN
MODEL_TYPE = MODEL_TYPE_LENET5    # For LeNet-5
MODEL_TYPE = MODEL_TYPE_ALEXNET   # For AlexNet
MODEL_TYPE = MODEL_TYPE_VGGNET    # For VGGNet
```

### 4. Outputs
- **Training History**: Graphs of accuracy and loss for both training and validation phases.
- **Confusion Matrix**: Visual representation of model predictions.
- **Classification Report**: Detailed precision, recall, and F1-score metrics.

### 5. Resuming from a Saved Model
If a saved model exists for the selected architecture, it will be automatically loaded. Otherwise, a new model will be trained, evaluated, and saved.

---

## Model Architectures

### 1. **Convolutional Neural Network (CNN)**
- Lightweight and fast to train.
- Ideal for real-time applications with near-perfect accuracy.

### 2. **LeNet-5**
- A classic architecture with low computational cost.
- Suited for small datasets or systems with limited resources.

### 3. **AlexNet**
- High accuracy but computationally expensive.
- Best for applications requiring precision over speed.

### 4. **VGGNet**
- Highly accurate with deep layers.
- Resource-intensive, suitable for high-performance systems.

---

## Results
The project evaluates each model based on:
- **Test Accuracy**: Overall model performance on unseen data.
- **Misclassifications**: Insights from confusion matrices.
- **Training History**: Visual trends of model learning.

Example outputs:
- Training history (`training_history_<model>.png`)
- Confusion matrix (`confusion_matrix_<model>.png`)
- Evaluation report (`report_model_<model>.txt`)

---

## Future Enhancements
- **Data Augmentation**: Introduce techniques like flipping, rotation, and scaling to improve generalization.
- **Real-World Testing**: Deploy the model on embedded systems (e.g., Raspberry Pi) for gesture recognition in real-time.
- **Additional Models**: Explore architectures like ResNet or MobileNet for better performance.

---

## License
This project is distributed under the MIT License.

---

**Author**: Donny Marthen Sitompul  
**Course**: Artificial Intelligence for Engineers (DAT305)
