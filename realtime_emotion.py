import sys
sys.setrecursionlimit(10000)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Load models
gender_net = cv2.dnn.readNetFromCaffe("deploy_gender.prototxt", "gender_net.caffemodel")
age_net = cv2.dnn.readNetFromCaffe("deploy_age.prototxt", "age_net.caffemodel")
emotion_model = load_model('emotion_model.h5')
gesture_model = load_model('asl_gesture_model.h5')

# Labels
gender_list = ['Male', 'Female']
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
gesture_labels = [chr(ord('A') + i) for i in range(26)]  # A-Z

# Face detector
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Word-building variables
current_word = ""
sentence = ""
last_gesture = None
last_gesture_time = time.time()
gesture_debounce_time = 1.0  # seconds between same gestures
word_timeout = 2.0  # seconds to finalize a word

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    (h, w) = frame.shape[:2]

    # Face Detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.5:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        face_roi = frame[startY:endY, startX:endX]

        # Emotion Detection
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 48, 48, 1))
        emotion_pred = emotion_model.predict(reshaped)
        emotion = emotions[np.argmax(emotion_pred)]

        # Age and Gender Detection
        face_blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227),
                                          (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        gender_net.setInput(face_blob)
        gender = gender_list[np.argmax(gender_net.forward())]

        age_net.setInput(face_blob)
        age = age_list[np.argmax(age_net.forward())]

        # Draw face info
        label = f"{emotion}, {gender}, {age}"
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Hand Gesture Recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks).reshape(1, -1)

            prediction = gesture_model.predict(landmarks)
            gesture_idx = np.argmax(prediction)

            if 0 <= gesture_idx < len(gesture_labels):
                gesture = gesture_labels[gesture_idx]
            else:
                gesture = "Unknown"

            # Debounce logic
            if gesture != last_gesture or (current_time - last_gesture_time) > gesture_debounce_time:
                if gesture != "Unknown":
                    current_word += gesture
                    last_gesture = gesture
                    last_gesture_time = current_time

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Auto-finalize word after timeout
    if current_word and (current_time - last_gesture_time) > word_timeout:
        sentence += current_word + " "
        current_word = ""

    # Display text on screen
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Gesture: {gesture if 'gesture' in locals() else 'None'}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Current Word: {current_word}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Sentence: {sentence + current_word}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Real-Time Multi-Feature Detection', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):  # Clear sentence
        sentence = ""
        current_word = ""

cap.release()
cv2.destroyAllWindows()