import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("model/sign_language_model.h5")

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)


# Define class labels (A-Z)
class_labels = sorted(os.listdir("dataset"))

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for better user experience
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract hand region
            x_min = y_min = float('inf')
            x_max = y_max = float('-inf')
            
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                x_min, y_min = min(x, x_min), min(y, y_min)
                x_max, y_max = max(x, x_max), max(y, y_max)

            # Crop and preprocess the hand region
            if x_max - x_min > 0 and y_max - y_min > 0:
                hand_img = frame[y_min:y_max, x_min:x_max]
                hand_img = cv2.resize(hand_img, (64, 64))
                hand_img = np.expand_dims(hand_img, axis=0) / 255.0  # Normalize

                # Predict gesture
                prediction = model.predict(hand_img)
                label = class_labels[np.argmax(prediction)]

                # Display prediction
                cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show output
    cv2.imshow("Sign Language Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
