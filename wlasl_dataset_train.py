import os
import cv2
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tqdm import tqdm

# --------------------------
# Parameters
# --------------------------
IMAGE_SIZE = 96        # Smaller resolution
NUM_FRAMES = 16        # Fewer frames per video
BATCH_SIZE = 4         # Small batch for memory safety
EPOCHS = 10
DATA_DIR = "WLASL"  # Path to WLASL dataset
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
JSON_PATH = os.path.join(DATA_DIR, "WLASL_v0.3.json")

# --------------------------
# Load JSON and Prepare Data
# --------------------------
def load_data():
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    samples = []
    labels = []

    label_set = set()

    for item in data:
        gloss = item['gloss']
        for inst in item['instances']:
            video_id = inst['video_id']
            video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
            if os.path.exists(video_path):
                samples.append(video_path)
                labels.append(gloss)
                label_set.add(gloss)

    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    num_classes = len(label_set)

    return samples, encoded_labels, le.classes_, num_classes

# --------------------------
# Video Frame Loader
# --------------------------
def video_generator(video_paths, labels, num_frames=NUM_FRAMES):
    for path, label in zip(video_paths, labels):
        yield load_video_frames(path, num_frames), label

# Dataset creation moved to main()
def load_video_frames(path, n_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(path)
    frames = []

    while len(frames) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            # Resize and normalize
            frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = (frame / 255.0).astype(np.float32)  # Explicit float32
            frames.append(frame)
        except Exception as e:
            print(f"Error processing frame from {path}: {e}")
            continue

    cap.release()

    # Pad with zeros if fewer frames
    while len(frames) < n_frames:
        frames.append(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32))
    frames = frames[:n_frames]  # Truncate if longer

    return np.array(frames, dtype=np.float32)  # Final array as float32

# --------------------------
# Build Model
# --------------------------
def build_model(input_shape, num_classes):
    base_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base_model.trainable = False

    cnn = tf.keras.Sequential([
        layers.TimeDistributed(base_model),
        layers.TimeDistributed(layers.GlobalAveragePooling2D())
    ])

    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        cnn,
        layers.LSTM(128, return_sequences=False),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
# --------------------------
# Main Training Pipeline
# --------------------------

def main():
    print("Loading data...")
    video_paths, labels, _, num_classes = load_data()

    print("Creating dataset pipeline...")
    dataset = tf.data.Dataset.from_generator(
        lambda: video_generator(video_paths, labels),
        output_types=(tf.float32, tf.int64),
        output_shapes=((NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3), ())
    )

    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    print("Building model...")
    model = build_model((NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3), num_classes)

    print("Training model...")
    model.fit(dataset, epochs=EPOCHS)

    print("Saving model...")
    model.save("wlasl_gesture_model.h5")

if __name__ == "__main__":
    main()