# Gesture Recognition with CNN

This project performs gesture recognition using a Convolutional Neural Network (CNN) model on a custom dataset of grayscale images. The dataset is structured in folders, each representing a unique gesture, and the project involves preprocessing the images, training a CNN model, and evaluating it with multiple metrics.

## Table of Contents

- [Dataset Structure](#dataset-structure)
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [License](#license)

## Dataset Structure

The dataset is organized in the following structure:

```
input/
└── leapGestRecog/
    ├── 00/ - Gesture folder 0
    ├── 01/ - Gesture folder 1
    ├── ...
    └── 09/ - Gesture folder 9
```

Each gesture folder contains grayscale images of a specific gesture. Each gesture is associated with a unique folder name used to generate lookup codes.

## Project Overview

1. **Image Preprocessing**: 
   - Images are resized and converted to grayscale.
   - Labels are assigned to each gesture based on the folder structure.
   
2. **CNN Model Training**: 
   - A CNN model is created with three convolutional layers and trained using the preprocessed images and gesture labels.

3. **Evaluation**:
   - Evaluation metrics include Accuracy, Confusion Matrix, Precision, Recall, and F1-score.
   - Training and validation accuracy and loss are plotted over epochs.

## Installation

To set up the project, ensure you have the required dependencies installed:

```bash
pip install numpy pillow matplotlib seaborn scikit-learn tensorflow
```

## Usage

1. Clone the repository and set up the dataset as described above.

2. Run the main code to train and evaluate the model:

```bash
python gesture_recognition.py
```

3. The script will:
   - Preprocess the images.
   - Train the CNN model.
   - Display training/validation accuracy and loss over epochs.
   - Display evaluation metrics including the confusion matrix and classification report.

## Model Architecture

The CNN model used in this project has the following layers:

- **Conv2D**: 32 filters, kernel size (5, 5), activation `ReLU`, with max pooling.
- **Conv2D**: 64 filters, kernel size (3, 3), activation `ReLU`, with max pooling.
- **Conv2D**: 64 filters, kernel size (3, 3), activation `ReLU`, with max pooling.
- **Flatten**
- **Dense**: 128 neurons, activation `ReLU`
- **Dense**: 10 neurons, activation `softmax`

The model uses categorical cross-entropy as the loss function and `rmsprop` as the optimizer.

## Evaluation Metrics

1. **Accuracy**: Measures the percentage of correctly predicted gestures.
2. **Confusion Matrix**: Visual representation of model predictions vs. actual labels.
3. **Precision, Recall, and F1-Score**: Evaluation metrics for each class, providing insights into model performance beyond accuracy.

### Visualizations

The training script also provides visualizations:
- **Training and Validation Accuracy**: Plotted over epochs.
- **Training and Validation Loss**: Plotted over epochs.
- **Confusion Matrix**: Visualized using a heatmap.

## Results

After training, the model achieves the following results:
- **Accuracy**: Test set accuracy is displayed after evaluation.
- **Classification Report**: Precision, recall, and F1-score for each gesture.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
```

