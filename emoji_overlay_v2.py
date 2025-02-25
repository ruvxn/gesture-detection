import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import time

# Load Hand Gesture Model V3 updated in gesture_detection.py
model = tf.keras.models.load_model("gesture_model_v3.h5")

# Initialize MediaPipe Hands and Drawing Utilities - these will be used for hand detection and visualization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Selfie Segmentation - this will be used for background segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)


# the media pipe hands model is used to detect the hand landmarks in the frame which will be used to classify the hand gesture
# the media pipe selfie segmentation model is used to segment the background from the frame which will help in replacing the background with a different image based on the hand gesture detected

backgrounds = {
    "thumbsup": cv2.imread("thumbsup_bg.png"),
    "thumbsdown": cv2.imread("thumbsdown_bg.png"),
    "peace": cv2.imread("peace_bg.png"),
    "love": cv2.imread("love_bg.png")
}


cap = cv2.VideoCapture(0)

last_predicted_label = None # keeps track of last predicted label 

while cap.isOpened(): # loop to capture frames from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    #preprocess the frame
    frame = cv2.flip(frame, 1) 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Background Segmentation using MediaPipe Selfie Segmentation
    segmentation_results = selfie_segmentation.process(rgb_frame)
    mask = segmentation_results.segmentation_mask > 0.5 
    mask = (mask * 255).astype(np.uint8)

    # Process Hands using MediaPipe Hands
    results = hands.process(rgb_frame)

    predicted_label = "No Hand Detected" # default no hand detected 
    selected_background = None # default no background 


# results.multi_hand_landmarks is used to check if the model has detected any hands in the frame
# if hands are detected, the bounding box coordinates are calculated to crop the hand region for classification
# the hand region is then resized and preprocessed for classification using the trained model
# the predicted label is then used to select the background image based on the hand gesture detected
    if results.multi_hand_landmarks: 
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape 
            x_min, y_min, x_max, y_max = w, h, 0, 0

            for landmark in hand_landmarks.landmark: 
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            margin = 30
            x_min, y_min = max(x_min - margin, 0), max(y_min - margin, 0)
            x_max, y_max = min(x_max + margin, w), min(y_max + margin, h)
            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.shape[0] > 0 and hand_img.shape[1] > 0:
                hand_img = cv2.resize(hand_img, (224, 224))
                hand_img = img_to_array(hand_img) / 255.0
                hand_img = np.expand_dims(hand_img, axis=0)

                predictions = model.predict(hand_img, verbose=0) # prediction is made using the trained model
                predicted_class = np.argmax(predictions)
                confidence = predictions[0][predicted_class] # confidence score for the predicted class

                class_labels = {0: "love", 1: "peace", 2: "thumbsdown", 3: "thumbsup"}
                predicted_label = f"{class_labels[predicted_class]} ({confidence * 100:.2f}%)" # display the predicted label and confidence score

                if confidence > 0.85: # a prerequisite for showing the background image is that the confidence score should be greater than 85%. This will ensure that the background doesn't change randomly
                    selected_background = backgrounds[class_labels[predicted_class]]
                    last_predicted_label = class_labels[predicted_class]

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, predicted_label, (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Apply Background Replacement if a hand gesture is detected
    if selected_background is not None:
        selected_background = cv2.resize(selected_background, (frame.shape[1], frame.shape[0]))

        mask_inv = cv2.bitwise_not(mask)
        body_part = cv2.bitwise_and(frame, frame, mask=mask)
        background_part = cv2.bitwise_and(selected_background, selected_background, mask=mask_inv)
        
        frame = cv2.add(body_part, background_part)

    cv2.imshow("Hand Gesture Background Changer", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
