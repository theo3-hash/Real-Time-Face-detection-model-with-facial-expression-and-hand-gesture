import os
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN logs

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load ASL Dataset
def load_asl_dataset(dataset_path):
    images = []
    labels = []
    class_names = sorted(os.listdir(dataset_path))
    
    for class_name in class_names:
        class_path = os.path.join(dataset_path, class_name)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (28, 28))  # Resize to 28x28 for consistency
            images.append(image)
            labels.append(class_names.index(class_name))
    
    return np.array(images), np.array(labels)

# Load Sign Language MNIST Dataset
def load_mnist_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    if not os.access(csv_path, os.R_OK):
        raise PermissionError(f"No read permissions for {csv_path}")
    
    # Skip the header row (skiprows=1)
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    labels = data[:, 0]
    images = data[:, 1:].reshape(-1, 28, 28)
    return images, labels

# Load and combine datasets
asl_images, asl_labels = load_asl_dataset(r"C:\Users\Debasish\Desktop\project\ASL\asl_alphabet_train\asl_alphabet_train")

# Load Sign Language MNIST dataset
csv_path = r"C:\Users\Debasish\Desktop\project\MNIST\sign_mnist_train.csv"
if not os.path.exists(csv_path):
    print(f"Error: File not found at {csv_path}")
elif not os.access(csv_path, os.R_OK):
    print(f"Error: No read permissions for {csv_path}")
else:
    mnist_images, mnist_labels = load_mnist_dataset(csv_path)

    # Combine datasets
    images = np.concatenate((asl_images, mnist_images), axis=0)
    labels = np.concatenate((asl_labels, mnist_labels), axis=0)

    # Check for invalid labels
    print("Unique labels in the dataset:", np.unique(labels))
    invalid_labels = labels[(labels < 0) | (labels >= 26)]
    if len(invalid_labels) > 0:
        print(f"Warning: Found {len(invalid_labels)} invalid labels. Removing them.")
        valid_indices = (labels >= 0) & (labels < 26)
        images = images[valid_indices]
        labels = labels[valid_indices]

    # Normalize images
    images = images / 255.0

    # Reshape images for CNN input
    images = np.expand_dims(images, axis=-1)  # Add channel dimension

    # Convert labels to one-hot encoding
    num_classes = 26  # A-Z
    labels = to_categorical(labels, num_classes)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Build the model
    def create_gesture_model(input_shape=(28, 28, 1), num_classes=26):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        return model

    model = create_gesture_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=64,
        validation_data=(X_test, y_test)
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Save the model
    model.save('gesture_model.h5')