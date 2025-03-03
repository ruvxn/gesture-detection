# Copyright (c) 2024 Ruveen Jayasinghe
# Licensed under the Apache License, Version 2.0
import cv2 
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import collections
import time

# Load hand gesture model
model = tf.keras.models.load_model("gesture_model_v5.h5")

# Initialize MediaPipe hands and drawing utilities
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7) # Updated max_num_hands=2 for two hand gestures
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe selfie segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Video paths 
video_paths = {
    "thumbsup": "1.mov",
    "thumbsdown": "2.mov",
    "peace": "3.mov",
    "love": "4.mov",
    "both_thumbsup": "5.mov",  
    "both_thumbsdown": "6.mov" 
}
video_captures = {gesture: cv2.VideoCapture(path) for gesture, path in video_paths.items()} 

cap = cv2.VideoCapture(0)

current_video = None  # Keeps track of the currently playing video
video_playing = False # Check if a video is playing

# Set transparency level for blending the video background with the original frame
alpha = 0.5  

# Gesture detection history 
gesture_history = collections.deque(maxlen=15)
stable_gesture = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Background segmentation
    segmentation_results = selfie_segmentation.process(rgb_frame)
    mask = segmentation_results.segmentation_mask > 0.3
    mask = (mask * 255).astype(np.uint8)

    # Process hands using MediaPipe hands
    if not video_playing:
        results = hands.process(rgb_frame)
        predicted_label = "No Hand Detected"
        selected_video = None
        hand_predictions = []  # Store gestures detected - used for two hand gestures update

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                x_min, y_min, x_max, y_max = w, h, 0, 0

                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)

                # Hand Size & Position Check - Avoids accidental detections
                hand_width = x_max - x_min
                hand_height = y_max - y_min
                if hand_width < 50 or hand_height < 50:
                    continue  

                margin = 30
                x_min, y_min = max(x_min - margin, 0), max(y_min - margin, 0)
                x_max, y_max = min(x_max + margin, w), min(y_max + margin, h)
                hand_img = frame[y_min:y_max, x_min:x_max]

                if hand_img.shape[0] > 0 and hand_img.shape[1] > 0:
                    hand_img = cv2.resize(hand_img, (224, 224))
                    hand_img = img_to_array(hand_img) / 255.0
                    hand_img = np.expand_dims(hand_img, axis=0)

                    predictions = model.predict(hand_img, verbose=0)
                    predicted_class = np.argmax(predictions)
                    confidence = predictions[0][predicted_class]

                    class_labels = {0: "love", 1: "peace", 2: "thumbsdown", 3: "thumbsup"}
                    predicted_label = f"{class_labels[predicted_class]} ({confidence * 100:.2f}%)"

                    if confidence > 0.90:
                        hand_predictions.append(class_labels[predicted_class])
                    else:
                        hand_predictions.append("None")

                # Draw bounding box and label
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, predicted_label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Identify two hand gestures for two hands detected simultaneously 
        if len(hand_predictions) == 2:
            if hand_predictions[0] == "thumbsup" and hand_predictions[1] == "thumbsup":
                stable_gesture = "both_thumbsup"
            elif hand_predictions[0] == "thumbsdown" and hand_predictions[1] == "thumbsdown":
                stable_gesture = "both_thumbsdown"
            else:
                stable_gesture = None
        else:
            stable_gesture = hand_predictions[0] if len(hand_predictions) == 1 else None

        # Store the detected gesture in history
        if stable_gesture is not None:
            gesture_history.append(stable_gesture)
        else:
            gesture_history.append("None")

        # Check if the same gesture was detected for 15 frames for more accurate detection
        if len(set(gesture_history)) == 1 and gesture_history[0] != "None":
            selected_video = gesture_history[0]

            if selected_video in video_captures and not video_playing:
                current_video = video_captures[selected_video]
                current_video.set(cv2.CAP_PROP_POS_FRAMES, 0)  
                video_playing = True  

    # Play video background with transparency handling
    if video_playing and current_video is not None:
        ret, video_frame = current_video.read()
        if not ret:
            video_playing = False 
            current_video = None
        else:
            video_frame = cv2.resize(video_frame, (frame.shape[1], frame.shape[0]))

            if video_frame.shape[-1] == 4:  
                b, g, r, a = cv2.split(video_frame)
                alpha_mask = cv2.merge([a, a, a])
                alpha_mask = alpha_mask / 255.0
                video_frame = cv2.merge([b, g, r])

                frame = ((frame * (1 - alpha)) + (video_frame * alpha)).astype(np.uint8)

            else:
                mask_inv = cv2.bitwise_not(mask)
                body_part = cv2.bitwise_and(frame, frame, mask=mask)
                background_part = cv2.bitwise_and(video_frame, video_frame, mask=mask_inv)

                blended_background = cv2.addWeighted(background_part, alpha, frame, 1 - alpha, 0)
                frame = cv2.add(body_part, blended_background)

    cv2.imshow("Hand Gesture Video Background", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
for capture in video_captures.values():
    capture.release()
cv2.destroyAllWindows()
