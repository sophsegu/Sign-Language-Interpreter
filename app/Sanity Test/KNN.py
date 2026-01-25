import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.neighbors import KNeighborsClassifier
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")

x = []
y = []

path = r"C:\Users\sophs\OneDrive\Desktop\Sign-Language-Interpreter\asl_data"
dir_list = os.listdir(path)

for item in dir_list:
    second_path = os.path.join(path, item)
    for file in os.listdir(second_path):
        y.append(item)
        x.append(np.load(os.path.join(second_path, file)))

# Checked if index size matches the label size. (Also previoussly checked that landmarks are of correct size)
#if len(x) == len(y):
    #print("Data loaded successfully.")

knn = KNeighborsClassifier(n_neighbors=15)

knn.fit(x, y)

def normalize_landmarks(hand):
    points = np.array([[lm.x, lm.y, lm.z] for lm in hand])
    origin = points[0]  # wrist
    points -= origin
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0:
        points /= scale
    return points.flatten()  # 63-d vector


base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    # Draw landmarks for feedback
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        for lm in hand:
            h, w, _ = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        features = normalize_landmarks(hand)
        prediction = knn.predict([features])
        cv2.putText(frame, f"Prediction: {prediction[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame **after all drawing**
    cv2.imshow("Hand Tracking - Data Collection", frame)

    # Break on ESC key
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
