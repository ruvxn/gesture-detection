
# Gesture Recognition with Dynamic Background & Emoji Overlays

## 📌 Project Overview
This project enables **real-time hand gesture recognition** using **MediaPipe, TensorFlow, and OpenCV**, with additional **gesture-based emoji overlays and video background changes**. Users can train their own models and test them with **interactive visual effects**.

### 🔥 Features
- **Real-time Hand Gesture Recognition** using Deep Learning.
- **Collect & Train Custom Gestures** with an easy-to-use pipeline.
- **Emoji Overlays** that appear dynamically based on detected gestures.
- **Video Background Switching** triggered by specific gestures.
- **Supports Single & Two-Hand Gestures** for a richer interactive experience.

---

## 📌 Version History
| Version | Features & Updates | File Name |
|---------|--------------------|-----------|
| v1.0    | Basic Gesture Collection & Classification | `collect_gestures.py`, `gesture_detection.py` |
| v2.0    | Real-time Gesture Detection & Classification | `dep_model.py` |
| v3.0    | Gesture-Based Emoji Overlay | `emoji_overlay_v1.py` |
| v3.1    | Improved Emoji Overlay Effects | `emoji_overlay_v2.py` |
| v4.0    | Gesture-Based Background Replacement | `background_v1.py` |
| v5.0    | Final Version - Gesture-Based Video Backgrounds & Multi-Hand Detection | `background_v2.py` |

---

## 📌 Setup Guide
This section will help you **collect gestures, train a model, and test it** using this project.

### **1️⃣ Install Dependencies**
Ensure you have Python installed, then install the required libraries:
```bash
pip install opencv-python mediapipe tensorflow numpy matplotlib
```

### **2️⃣ Collect Custom Gesture Data**
To create your own dataset:
```bash
python collect_gestures.py
```
- A webcam will open.
- Select the gesture class you are collecting images for (options will be displayed).
- **Press 's'** to save cropped hand images.
- **Press 'q'** to quit after collecting enough data.
- Images are saved in the `cropped_gesture_dataset/` folder.

### **3️⃣ Train Your Own Model**
Use the Jupyter Notebook:
```bash
jupyter notebook gesture_detection.ipynb
```
- Update `DATASET_PATH` in the notebook to your dataset folder.
- Train a new model with **MobileNetV2**.
- Save the trained model as `gesture_model.h5`.

### **4️⃣ Test Real-Time Gesture Recognition**
After training, test your model using:
```bash
python dep_model.py
```
- This will **detect hand gestures and display them on screen**.

### **5️⃣ Run Emoji Overlay Program**
```bash
python emoji_overlay_v2.py
```
- When a recognized gesture is detected, an **emoji appears dynamically**.

### **6️⃣ Run Gesture-Based Video Background Program**
```bash
python background_v2.py
```
- The **background changes** based on detected gestures.
- **Supports two-hand gestures** for advanced interactions.

---

## 📌 Notes & Recommendations
- Ensure **good lighting** for best gesture recognition.
- The **gesture collection process** should have a **consistent background**.
- The **model improves with more diverse training data**.



