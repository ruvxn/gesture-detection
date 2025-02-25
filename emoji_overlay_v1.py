import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import time

model = tf.keras.models.load_model("gesture_model_v3.h5")

# Load mediapipe hands model and drawing utilities - these will be used for hand detection and visualization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) # - number of hands is set to be one for this model because all the gestures trained are single handed.
                                                                                               # - the minimum detection confidence is set to 0.5 to ensure that the model is confident in the detection of the hand
                                                                                               #   which will help in reducing false positives and negatives.

mp_drawing = mp.solutions.drawing_utils # - this is used to draw the landmarks on the hand detected by the model

# Load emoji images with alpha channel - alpha channel is used to control the transparency of the image when overlayed on the frame
emoji_images = {
    "thumbsup": cv2.imread("thumbsup.png", cv2.IMREAD_UNCHANGED),
    "thumbsdown": cv2.imread("thumbsdown.png", cv2.IMREAD_UNCHANGED),
    "peace": cv2.imread("peace.png", cv2.IMREAD_UNCHANGED),
    "love": cv2.imread("love.png", cv2.IMREAD_UNCHANGED)
}


cap = cv2.VideoCapture(0)

emoji_opacity = 0  # this is used to control the transparency of the image
emoji_scale = 0.8  # this is used for the zoom in effect
last_emoji_time = time.time() # this is used to keep track of the time when the last emoji was shown - this is used to control the animation, so that the emoji is not shown for a long time
last_predicted_label = None

def overlay_emoji(frame, emoji, x, y, scale=1.0, alpha=1.0): # this function is used to overlay the emoji on the frame
    if emoji is None:
        return frame

    # Resize emoji based on scale
    size = int(150 * scale)  
    emoji = cv2.resize(emoji, (size, size), interpolation=cv2.INTER_AREA)  # ERROR FIXED: cv2.INTER_AREA is used for resizing the image to a smaller size in order to reduce the aliasing effect

    # Apply alpha blending to overlay emoji on the frame
    if emoji.shape[2] == 4:  # Check if image has alpha channel (4th channel)
        alpha_channel = emoji[:, :, 3] / 255.0  # Normalize alpha values 
        color_channels = emoji[:, :, :3]  # Extract RGB channels 

        # Get dimensions and position
        h, w, _ = emoji.shape
        x, y = max(x - w // 2, 0), max(y - h // 2, 0)  # The center emoji on x, y coordinates 

        # Ensure emoji fits within frame boundaries 
        if x + w > frame.shape[1]:
            w = frame.shape[1] - x
            emoji = emoji[:, :w]
            alpha_channel = alpha_channel[:, :w]
            color_channels = color_channels[:, :w]
        if y + h > frame.shape[0]:
            h = frame.shape[0] - y
            emoji = emoji[:h, :]
            alpha_channel = alpha_channel[:h, :]
            color_channels = color_channels[:h, :]

        # Blend emoji onto frame using alpha channel
        for c in range(3):  # Apply alpha blending per channel 
            frame[y:y+h, x:x+w, c] = (1 - alpha_channel) * frame[y:y+h, x:x+w, c] + alpha_channel * color_channels[:, :, c] # ERROR FIXED: The alpha blending formula is corrected to blend the emoji with the frame

    return frame

while cap.isOpened(): # this loop is used to capture the frames from the webcam 
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)  # Flip for mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert BGR to RGB for MediaPipe
    
    # Process hands in the frame and get hand landmarks 
    results = hands.process(rgb_frame)

    predicted_label = "No Hand Detected" 
    emoji_to_show = None
    emoji_x, emoji_y = 100, 100  # Default position of emoji

    if results.multi_hand_landmarks:                        # this condition checks if there are any hands detected in the frame 
        for hand_landmarks in results.multi_hand_landmarks: # this loop iterates over all the hands detected in the frame 
            h, w, _ = frame.shape                           # this gets the height and width of the frame in order to calculate the bounding box coordinates for the hand    
            x_min, y_min, x_max, y_max = w, h, 0, 0         # this initialises the bounding box coordinates for the hand which will be used to crop the hand region from the frame to perform the classification

            # Get hand bounding box coordinates
            for landmark in hand_landmarks.landmark:        # this loop iterates over all the landmarks detected on the hand to calculate the bounding box coordinates by finding the minimum and maximum x and y coordinates of the landmarks
                x, y = int(landmark.x * w), int(landmark.y * h) 
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)
                                                            
            # Expand bounding box slightly for better cropping and classification. because the landmarks are not exactly at the edges of the hand, and slightly expanded bounding box includes more of the hand region
            margin = 30  
            x_min, y_min = max(x_min - margin, 0), max(y_min - margin, 0)
            x_max, y_max = min(x_max + margin, w), min(y_max + margin, h)

            # Crop hand region for classification
            hand_img = frame[y_min:y_max, x_min:x_max]

         
            if hand_img.shape[0] > 0 and hand_img.shape[1] > 0: # Check if the cropped image is valid before classification by checking if the height and width of the image are greater than 0
                hand_img = cv2.resize(hand_img, (224, 224))     # Resize the cropped image to 224x224 as the model gesture_detection.py was trained on images of this size
                hand_img = img_to_array(hand_img) / 255.0  # image to arrat for preprocessing and then normalise the pixel values to be between 0 and 1
                hand_img = np.expand_dims(hand_img, axis=0)

                # PREDICTION
                predictions = model.predict(hand_img, verbose=0)  #performs prediction 
                predicted_class = np.argmax(predictions)         # get the index of the class with the highest probability
                confidence = predictions[0][predicted_class]    # get the probability of the predicted class which will be used as the confidence score

                # Class mapping
                class_labels = {0: "love", 1: "peace", 2: "thumbsdown", 3: "thumbsup"} # claa labels
                predicted_label = f"{class_labels[predicted_class]} ({confidence * 100:.2f}%)" #display label and confidence score

                # Emoji effects only if confidence is higher than 85%, This will ensure that the emoji is only shown when the model is confident enough that prediction is accurate
                if confidence > 0.85:
                    emoji_to_show = emoji_images[class_labels[predicted_class]]
                    emoji_x, emoji_y = x_max, y_min  # Position emoji near hand

                    # Reset animation state and scale if new gesture detected 
                    if class_labels[predicted_class] != last_predicted_label:
                        last_predicted_label = class_labels[predicted_class]
                        emoji_opacity = 0
                        emoji_scale = 0.8  
                        last_emoji_time = time.time()

                # Draw bounding box and label on the frame of the hand detected
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, predicted_label, (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw hand landmarks on the frame 
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Apply animation effects to emoji
    if emoji_to_show is not None:
        elapsed_time = time.time() - last_emoji_time

        # Scale animation for zoom effect
        if emoji_scale < 1.4:  
            emoji_scale += 0.07
        
        # Fade in effect by adjusting opacity
        if emoji_opacity < 1:
            emoji_opacity += 0.07

        # Floating effect by moving emoji upwards
        emoji_y -= int(elapsed_time * 6)

        # Overlay emoji with animations
        frame = overlay_emoji(frame, emoji_to_show, emoji_x, emoji_y, scale=emoji_scale, alpha=emoji_opacity)

    # Display output
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
