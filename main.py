import os
from typing import Dict, Tuple
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from keras import layers, models

def create_lookup(directory: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Creates lookup and reverse lookup dictionaries from folder names."""
    lookup = {name: i for i, name in enumerate(os.listdir(directory)) if not name.startswith('.')}
    reverselookup = {i: name for name, i in lookup.items()}
    return lookup, reverselookup

def load_images_for_gesture(gesture_path: str, image_size: Tuple[int, int] = (320, 120)) -> np.ndarray:
    """Loads and preprocesses images from a specified path."""
    return np.array([
        np.array(Image.open(os.path.join(gesture_path, image_file)).convert('L').resize(image_size), dtype='float32')
        for image_file in filter(lambda f: not f.startswith('.'), os.listdir(gesture_path))
    ])

def create_dataset(lookup: Dict[str, int], base_path: str = "input/leapGestRecog/") -> Tuple[np.ndarray, np.ndarray]:
    """Generates x_data and y_data arrays from images and lookup codes."""
    x_data, y_data = [], []
    for i in range(10):
        folder_path = os.path.join(base_path, f"0{i}")
        for gesture_folder in filter(lambda x: not x.startswith('.'), os.listdir(folder_path)):
            gesture_path = os.path.join(folder_path, gesture_folder)
            gesture_images = load_images_for_gesture(gesture_path)
            x_data.extend(gesture_images)
            y_data.extend([lookup[gesture_folder]] * len(gesture_images))
    return np.array(x_data), np.array(y_data).reshape(-1, 1)

def plot_sample_images(x_data: np.ndarray, y_data: np.ndarray, reverselookup: Dict[int, str], sample_count: int = 10) -> None:
    """Displays a sample of images with their labels."""
    for i in range(sample_count):
        plt.imshow(x_data[i * 200], cmap='gray')
        plt.title(reverselookup[y_data[i * 200, 0]])
        plt.axis('off')
        plt.show()

def prepare_data(x_data: np.ndarray, y_data: np.ndarray) -> Tuple:
    """Encodes labels, reshapes images, and splits data into training, validation, and test sets."""
    y_data = to_categorical(y_data)
    x_data = x_data.reshape(-1, 120, 320, 1)  # Adding channel dimension for grayscale
    x_train, x_temp, y_train, y_temp = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    x_validate, x_test, y_validate, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
    return x_train, x_validate, x_test, y_train, y_validate, y_test

def build_model(input_shape: Tuple[int, int, int]) -> models.Sequential:
    """Builds and compiles a Convolutional Neural Network model."""
    model = models.Sequential([
        layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_training_history(history):
    """Plots training and validation accuracy and loss over epochs."""
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, x_test, y_test, y_true_labels, reverselookup):
    """Evaluates the model and prints additional metrics like confusion matrix, precision, recall, and F1-score."""
    # Evaluate model accuracy
    loss, acc = model.evaluate(x_test, y_test, verbose=1)
    print("Test Accuracy:", acc)

    # Predict class labels
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[reverselookup[i] for i in range(10)], 
                yticklabels=[reverselookup[i] for i in range(10)])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

    # Display classification report
    report = classification_report(y_true, y_pred, target_names=[reverselookup[i] for i in range(10)])
    print("\nClassification Report:\n", report)

if __name__ == "__main__":
    directory = "input/leapGestRecog/00/"
    lookup, reverselookup = create_lookup(directory)
    print("Lookup:", lookup)
    print("Reverse Lookup:", reverselookup)

    # Load dataset and plot samples
    x_data, y_data = create_dataset(lookup)
    plot_sample_images(x_data, y_data, reverselookup)

    # Prepare data and split into training, validation, and test sets
    x_train, x_validate, x_test, y_train, y_validate, y_test = prepare_data(x_data, y_data)

    # Build and train the model
    model = build_model(input_shape=(120, 320, 1))
    history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_validate, y_validate), verbose=1)

    # Plot the training history
    plot_training_history(history)

    # Evaluate model and display metrics
    evaluate_model(model, x_test, y_test, y_data, reverselookup)
