import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Hands model and drawing utilities for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Set up folders for saving cropped hand images for each gesture 
DATASET_PATH = "cropped_gesture_dataset"  
GESTURES = ["thumbsup", "thumbsdown", "peace", "love"]
SAVE_PATH = os.path.join(os.getcwd(), DATASET_PATH)

# Create dataset directories if they don’t exist already
for gesture in GESTURES:
    os.makedirs(os.path.join(SAVE_PATH, gesture), exist_ok=True)

# Open webcam for real-time hand gesture collection
cap = cv2.VideoCapture(0)

print(" Press 's' to save a cropped image 'q' to quit.")

selected_gesture = None  # Store the selected class once

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip image for natural feel 
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB for MediaPipe 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box
            h, w, c = frame.shape
            x_min, y_min, x_max, y_max = w, h, 0, 0
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            # Expand the bounding box slightly
            margin = 20
            x_min = max(x_min - margin, 0)
            y_min = max(y_min - margin, 0)
            x_max = min(x_max + margin, w)
            y_max = min(y_max + margin, h)

            # Crop and display the hand region for better visualization
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.shape[0] > 0 and hand_img.shape[1] > 0:
                hand_resized = cv2.resize(hand_img, (224, 224))
                cv2.imshow("Cropped Hand", hand_resized)

                #  user select gesture class only once 
                if selected_gesture is None:
                    print(" Select a class: 1 (thumbsup), 2 (thumbsdown), 3 (peace), 4 (love)")
                    key = cv2.waitKey(0) & 0xFF
                    if key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                        selected_gesture = GESTURES[int(chr(key)) - 1]
                        print(f"Saving images as '{selected_gesture}' (Press 's' to save, 'q' to quit)")

                # Save image when 's' is pressed
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s') and selected_gesture is not None:
                    img_path = os.path.join(SAVE_PATH, selected_gesture, f"{selected_gesture}_{len(os.listdir(os.path.join(SAVE_PATH, selected_gesture)))}.jpg")
                    cv2.imwrite(img_path, hand_resized)
                    print(f" Image saved: {img_path}")

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display output
    cv2.imshow("Hand Detection", frame)
    
    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
