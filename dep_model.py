import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained model from gesture_detection.py
model = tf.keras.models.load_model("gesture_model_v2.h5")

# Load MediaPipe Hands model and drawing utilities - these will be used for hand detection and visualization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open webcam for real-time hand gesture recognition
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame horizontally for a mirrored effect 
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process hands in the frame and get hand landmarks
    results = hands.process(rgb_frame)

    # Default label if no hand is detected in the frame
    predicted_label = "No Hand Detected"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box coordinates for the hand 
            h, w, c = frame.shape
            x_min, y_min, x_max, y_max = w, h, 0, 0
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            # Expand the bounding box slightly for better cropping  
            margin = 20
            x_min = max(x_min - margin, 0)
            y_min = max(y_min - margin, 0)
            x_max = min(x_max + margin, w)
            y_max = min(y_max + margin, h)

            # Crop the hand region for classification
            hand_img = frame[y_min:y_max, x_min:x_max]

            # Ensure valid image size before classification 
            if hand_img.shape[0] > 0 and hand_img.shape[1] > 0:
                # Resize and preprocess for classification  
                hand_img = cv2.resize(hand_img, (224, 224))
                hand_img = img_to_array(hand_img) / 255.0
                hand_img = np.expand_dims(hand_img, axis=0)

                # Force reclassification on every frame - overcomes flickering issues 
                predictions = model.predict(hand_img, verbose=0)
                predicted_class = np.argmax(predictions)
                
                # Class mapping for display
                class_labels = {v: k for k, v in {"love": 0, "peace": 1, "thumbsdown": 2, "thumbsup": 3}.items()}
                predicted_label = class_labels[predicted_class]

                # Draw bounding box and label on the frame 
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, predicted_label, (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Display output
    cv2.imshow("Hand Gesture Recognition", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
