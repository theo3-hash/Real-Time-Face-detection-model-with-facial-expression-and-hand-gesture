import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Data storage
landmarks_data = []
labels = []

# Label mapping (A=0, B=1, ..., Z=25)
label_map = {chr(i): i - ord('A') for i in range(ord('A'), ord('Z') + 1)}

# Collect data for each gesture
gesture = 'A'  # Change this for each gesture
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])

            # Save landmarks and label
            landmarks_data.append(landmarks)
            labels.append(label_map[gesture])

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Collecting Data', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save data
np.save('landmarks_data.npy', np.array(landmarks_data))
np.save('labels.npy', np.array(labels))

cap.release()
cv2.destroyAllWindows()