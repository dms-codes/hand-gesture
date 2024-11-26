import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from keras import layers, models,Input
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import load_model
from keras.optimizers import Adam
import pickle
import io
from contextlib import redirect_stdout

MODEL_TYPE = "CNN"
MODEL_TYPE_CNN = "CNN"
MODEL_TYPE_LENET5 = "LENET5"
MODEL_TYPE_ALEXNET = "ALEXNET"
MODEL_TYPE_VGGNET = "VGGNET"

MODEL_PATH = 'gesture_model.h5'
MODEL_PATH_CNN = 'gesture_model_cnn.keras'
MODEL_PATH_LENET5 = 'gesture_model_lenet5.keras'
MODEL_PATH_ALEXNET = 'gesture_model_alexnet.keras'
MODEL_PATH_VGGNET = 'gesture_model_vggnet.keras'


CONFUSION_MATRIX_PATH_CNN ='confusion_matrix_cnn.png'
CONFUSION_MATRIX_PATH_LENET5 ='confusion_matrix_lenet5.png'
CONFUSION_MATRIX_PATH_ALEXNET ='confusion_matrix_alexnet.png'
CONFUSION_MATRIX_PATH_VGGNET ='confusion_matrix_vggnet.png'


REPORT_MODEL_CNN = "report_model_cnn.txt"
REPORT_MODEL_LENET5 = "report_model_lenet5.txt"
REPORT_MODEL_ALEXNET = "report_model_alexnet.txt"
REPORT_MODEL_VGGNET = "report_model_vggnet.txt"


LOOKUP_FILENAME = 'lookup.pickle'
DATASET_FILENAME = 'dataset.pickle'

TRAINING_HISTORY_CNN_FILENAME = "training_history_cnn.png"
TRAINING_HISTORY_LENET5_FILENAME = "training_history_lenet5.png"
TRAINING_HISTORY_ALEXNET_FILENAME = "training_history_alexnet.png"
TRAINING_HISTORY_VGGNET_FILENAME = "training_history_vggxnet.png"


def capture_model_summary(model):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        model.summary()
    summary = buffer.getvalue()
    buffer.close()
    return summary

def get_model_path(model_type):
    if model_type == MODEL_TYPE_CNN:
            return MODEL_PATH_CNN
    elif model_type == MODEL_TYPE_LENET5:
            return MODEL_PATH_LENET5
    elif model_type == MODEL_TYPE_ALEXNET:
            return MODEL_PATH_ALEXNET
    elif model_type == MODEL_TYPE_VGGNET:
            return MODEL_PATH_VGGNET
        
def get_report_filename(model):
    if model.type == MODEL_TYPE_CNN:
            return REPORT_MODEL_CNN
    elif model.type == MODEL_TYPE_LENET5:
            return REPORT_MODEL_LENET5
    elif model.type == MODEL_TYPE_ALEXNET:
            return REPORT_MODEL_ALEXNET
    elif model.type == MODEL_TYPE_VGGNET:
            return REPORT_MODEL_VGGNET
    
def get_confusion_matrix_filename(model):
    if model.type == MODEL_TYPE_CNN:
            return CONFUSION_MATRIX_PATH_CNN
    elif model.type == MODEL_TYPE_LENET5:
            return CONFUSION_MATRIX_PATH_LENET5
    elif model.type == MODEL_TYPE_ALEXNET:
            return CONFUSION_MATRIX_PATH_ALEXNET
    elif model.type == MODEL_TYPE_VGGNET:
            return CONFUSION_MATRIX_PATH_VGGNET

def get_training_history_plot_filename(model):
    if model.type == MODEL_TYPE_CNN:
        return TRAINING_HISTORY_CNN_FILENAME
    elif model.type == MODEL_TYPE_LENET5:
        return TRAINING_HISTORY_LENET5_FILENAME 
    elif model.type == MODEL_TYPE_ALEXNET:
        return TRAINING_HISTORY_ALEXNET_FILENAME 
    elif model.type == MODEL_TYPE_VGGNET:
        return TRAINING_HISTORY_VGGNET_FILENAME 
    
def create_lookup(directory: str):
    print("0. Creating Folder Structure and Lookup")
    if not os.path.exists(LOOKUP_FILENAME):
        with open(LOOKUP_FILENAME, 'ab') as resfile:
            lookup = {name: i for i, name in enumerate(os.listdir(directory)) if not name.startswith('.')}
            reverselookup = {i: name for name, i in lookup.items()}
            res = lookup, reverselookup
            pickle.dump(res, resfile)
            resfile.close()
        print(f"File '{LOOKUP_FILENAME}' created and data stored.")
    else:
        resfile = open(LOOKUP_FILENAME, 'rb')    
        res = pickle.load(resfile) 
        resfile.close()               
    return res

def load_images_for_gesture(gesture_path: str, image_size=(320, 120)):
    return np.array([
        np.array(Image.open(os.path.join(gesture_path, image_file)).convert('L').resize(image_size), dtype='float32')
        for image_file in filter(lambda f: not f.startswith('.'), os.listdir(gesture_path))
    ])

def create_dataset(lookup, base_path="input/leapGestRecog/"):
    print("1. Creating dataset")
    if not os.path.exists(DATASET_FILENAME):
        with open(DATASET_FILENAME, 'ab') as resfile:
            x_data, y_data = [], []
            for i in range(10):
                folder_path = os.path.join(base_path, f"0{i}")
                for gesture_folder in filter(lambda x: not x.startswith('.'), os.listdir(folder_path)):
                    gesture_path = os.path.join(folder_path, gesture_folder)
                    gesture_images = load_images_for_gesture(gesture_path)
                    x_data.extend(gesture_images)
                    y_data.extend([lookup[gesture_folder]] * len(gesture_images))
            dataset = np.array(x_data), np.array(y_data).reshape(-1, 1)
            pickle.dump(dataset, resfile)
            resfile.close()
        print(f"File '{DATASET_FILENAME}' created and data stored.")
    else:
        resfile = open(DATASET_FILENAME, 'rb')    
        dataset = pickle.load(resfile) 
        resfile.close()               
    return dataset

def prepare_data(x_data, y_data):
    y_data = to_categorical(y_data)
    x_data = x_data.reshape(-1, 120, 320, 1)  # Adding channel dimension for grayscale
    print("2. Splitting the dataset")
    x_train, x_temp, y_train, y_temp = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    x_validate, x_test, y_validate, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
    return x_train, x_validate, x_test, y_train, y_validate, y_test

def build_model(model_type,x_train, x_validate, x_test, y_train, y_validate, y_test,input_shape=(120, 320, 1),num_classes=10):
    if model_type == MODEL_TYPE_CNN:
        model = build_model_cnn(input_shape)
    elif model_type == MODEL_TYPE_LENET5:
        model = build_model_lenet5(input_shape,num_classes)
    elif model_type == MODEL_TYPE_ALEXNET:
        model = build_model_alexnet(input_shape,num_classes)
    elif model_type == MODEL_TYPE_VGGNET:
        model = build_model_vggnet(input_shape,num_classes)
    model.type = model_type
    model.x_train = x_train
    model.x_validate = x_validate
    model.x_test = x_test
    model.y_train = y_train
    model.y_validate = y_validate
    model.y_test = y_test
    model.lookup = lookup
    model.reverselookup = reverselookup
    return model

def build_model_cnn(input_shape):
    print("3. Building the CNN model.")
    model = models.Sequential([
        Input(shape=input_shape),  # Explicitly define the input layer
        layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.summary_report = capture_model_summary(model)
    return model


def build_model_lenet5(input_shape, num_classes):
    print("3. Building the LeNet-5 model.")

    model = models.Sequential([
        # Input layer
        Input(shape=input_shape),

        # Layer 1: Convolutional + Average Pooling
        layers.Conv2D(6, kernel_size=(5, 5), activation="tanh", padding="valid"),
        layers.AveragePooling2D(pool_size=(2, 2)),

        # Layer 2: Convolutional + Average Pooling
        layers.Conv2D(16, kernel_size=(5, 5), activation="tanh"),
        layers.AveragePooling2D(pool_size=(2, 2)),

        # Layer 3: Fully Connected (Flatten + Dense)
        layers.Flatten(),
        layers.Dense(120, activation="tanh"),

        # Layer 4: Fully Connected
        layers.Dense(84, activation="tanh"),

        # Output Layer
        layers.Dense(num_classes, activation="softmax")
    ])
    model.summary_report = capture_model_summary(model)
    return model

def build_model_alexnet(input_shape, num_classes):
    print("3. Building the AlexNet model...")
    
    model = models.Sequential([
        # Input layer
        Input(shape=input_shape),

        # Layer 1: Convolutional + Max Pooling
        layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation="relu", padding="valid"),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),

        # Layer 2: Convolutional + Max Pooling
        layers.Conv2D(256, kernel_size=(5, 5), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),

        # Layer 3: Convolutional
        layers.Conv2D(384, kernel_size=(3, 3), activation="relu", padding="same"),

        # Layer 4: Convolutional
        layers.Conv2D(384, kernel_size=(3, 3), activation="relu", padding="same"),

        # Layer 5: Convolutional + Max Pooling
        layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),

        # Flatten + Fully Connected Layers
        layers.Flatten(),
        layers.Dense(4096, activation="relu"),
        layers.Dropout(0.5),  # Dropout for regularization
        layers.Dense(4096, activation="relu"),
        layers.Dropout(0.5),

        # Output Layer
        layers.Dense(num_classes, activation="softmax")
    ])
    model.summary_report = capture_model_summary(model)
    return model

def build_model_vggnet(input_shape=(120,320,1), num_classes=10):
    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 2
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 3
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 4
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 5
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.summary_report = capture_model_summary(model)
    return model

def compile_model(model):
    if model.type == MODEL_TYPE_CNN:
        model.compile(optimizer='rmsprop', 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
    elif model.type == MODEL_TYPE_LENET5:
        model.compile(optimizer=Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
    elif model.type == MODEL_TYPE_ALEXNET:
        model.compile(
            optimizer=Adam(learning_rate=0.001),  # Adam optimizer
            loss="categorical_crossentropy",                # Loss function for multi-class classification
            metrics=["accuracy"]                            # Track accuracy during training
        )
    elif model.type == MODEL_TYPE_VGGNET:
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model):
    if model.type == MODEL_TYPE_CNN:
        history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_validate, y_validate), verbose=1)
    elif model.type == MODEL_TYPE_LENET5:
        history = model.fit(
            x_train, y_train,
            validation_data=(x_validate, y_validate),
            epochs=20,
            batch_size=64)
    elif model.type == MODEL_TYPE_ALEXNET:
        history = model.fit(
            x_train, y_train,
            validation_data=(x_validate, y_validate),
            epochs=20,
            batch_size=64
        )
    elif model.type == MODEL_TYPE_VGGNET:
        history = model.fit(x_train, y_train, 
                            validation_data=(x_validate, y_validate),
                            epochs=20, 
                            batch_size=32
                            )

    model.history = history
    plot_training_history(model)
    return model

def plot_training_history(model):
    filename = get_training_history_plot_filename(model)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(model.history.history['accuracy'], label='Training Accuracy')
    plt.plot(model.history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(model.history.history['loss'], label='Training Loss')
    plt.plot(model.history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{filename}')
    plt.close()



def evaluate_model(model):
    report_filename = get_report_filename(model)
    confusion_matrix_filename = get_confusion_matrix_filename(model)

    res = "Model Evaluation Summary:\n"
    
    # Capture model summary
    res += model.summary_report  # Concatenate the summary string

    loss, acc = model.evaluate(model.x_test, model.y_test, verbose=1)
    res += f"Test Accuracy: {acc}.\n"


    y_pred = np.argmax(model.predict(model.x_test), axis=1)
    y_true = np.argmax(model.y_test, axis=1)

    print("Generating Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[model.reverselookup[i] for i in range(10)], 
                yticklabels=[model.reverselookup[i] for i in range(10)])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(confusion_matrix_filename)
    plt.close()

    print("Generating classification report")
    report = classification_report(y_true, y_pred, target_names=[model.reverselookup[i] for i in range(10)])
    res += report

    with open(report_filename, 'w') as f:
        f.write(res)

if __name__ == "__main__":
    print("""
Artificial Intelligence for Engineers DAT305
Hand Gesture Recognition
By Donny Marthen Sitompul          
""")
    directory = "input/leapGestRecog/00/"
    lookup, reverselookup = create_lookup(directory)
    x_data, y_data = create_dataset(lookup)


    #MODEL_TYPE = MODEL_TYPE_CNN
    #MODEL_TYPE = MODEL_TYPE_LENET5
    #MODEL_TYPE = MODEL_TYPE_ALEXNET
    MODEL_TYPE = MODEL_TYPE_VGGNET

    MODEL_PATH = get_model_path(MODEL_TYPE)

    if MODEL_TYPE == MODEL_TYPE_LENET5 or MODEL_TYPE == MODEL_TYPE_ALEXNET or MODEL_TYPE == MODEL_TYPE_VGGNET:
        x_data = x_data/255.0
    x_train, x_validate, x_test, y_train, y_validate, y_test = prepare_data(x_data, y_data)
    print("Check if the model is already saved")
    if os.path.exists(MODEL_PATH):
        print("3. Loading saved trained model...")
        model = load_model(MODEL_PATH)
        model.type = MODEL_TYPE
        model.x_train = x_train
        model.x_validate = x_validate
        model.x_test = x_test
        model.y_train = y_train
        model.y_validate = y_validate
        model.y_test = y_test
        model.lookup = lookup
        model.reverselookup = reverselookup
        evaluate_model(model)
    else:
        print("Saved model not found. Building and training a new model...")
        model = build_model(MODEL_TYPE,x_train, x_validate, x_test, y_train, y_validate, y_test)

        print("4. Compiling the model")
        model = compile_model(model)   
        
        print("5. Training the model")
        model = train_model(model)   
        print("Saving the trained model")
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

        print("6. Evaluating the model.")
        evaluate_model(model)
        

