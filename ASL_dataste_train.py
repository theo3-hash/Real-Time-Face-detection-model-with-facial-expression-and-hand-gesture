import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Step 1: Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Paths
DATASET_DIR = "asl_alphabet_dataset"  # Path to the ASL Alphabet Dataset
OUTPUT_DIR = "processed_asl_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 2: Extract Hand Landmarks from Images
def extract_landmarks():
    print("Extracting hand landmarks...")
    for gesture_label in os.listdir(DATASET_DIR):
        gesture_dir = os.path.join(DATASET_DIR, gesture_label)
        output_gesture_dir = os.path.join(OUTPUT_DIR, gesture_label)
        os.makedirs(output_gesture_dir, exist_ok=True)

        print(f"Processing gesture: {gesture_label}")

        for file_name in os.listdir(gesture_dir):
            file_path = os.path.join(gesture_dir, file_name)
            image = cv2.imread(file_path)
            if image is None:
                continue

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb_image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract landmarks
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])

                    # Save landmarks to file
                    output_file = os.path.join(output_gesture_dir, file_name.replace(".jpg", ".npy"))
                    np.save(output_file, np.array(landmarks))
                    print(f"Saved {output_file}")

# Step 3: Load and Prepare Data
def load_data():
    print("Loading processed data...")
    labels = []
    data = []

    for label, gesture in enumerate(os.listdir(OUTPUT_DIR)):
        gesture_dir = os.path.join(OUTPUT_DIR, gesture)
        for file_name in os.listdir(gesture_dir):
            file_path = os.path.join(gesture_dir, file_name)
            landmarks = np.load(file_path)
            data.append(landmarks)
            labels.append(label)

    # Convert to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Normalize the data
    data = data / np.max(data)

    # One-hot encode labels
    labels = to_categorical(labels)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Step 4: Train the Model
def train_model(X_train, X_test, y_train, y_test):
    print("Training the model...")
    num_classes = len(os.listdir(OUTPUT_DIR))

    # Define the model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(63,)),  # Input shape matches the number of landmarks
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')  # Output layer for number of gestures
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test)
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Save the model
    model.save('asl_gesture_model.h5')
    print("Model saved as 'asl_gesture_model.h5'.")

# Main Execution
if __name__ == "__main__":
    # Step 1: Extract landmarks
    extract_landmarks()

    # Step 2: Load and prepare data
    X_train, X_test, y_train, y_test = load_data()

    # Step 3: Train the model
    train_model(X_train, X_test, y_train, y_test)