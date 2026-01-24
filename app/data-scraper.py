import json
import cv2
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

VIDEO_DIR = "archive/videos"
OUTPUT_DIR = "asl_data"
JSON_PATH = "archive/WLASL_v0.3.json"
FRAME_STRIDE = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_landmarks(hand):
    points = np.array([[lm.x, lm.y, lm.z] for lm in hand])
    origin = points[0]
    points -= origin
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0:
        points /= scale
    return points.flatten()

def save_sample(label, features):
    label_dir = os.path.join(OUTPUT_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    count = len(os.listdir(label_dir))
    np.save(os.path.join(label_dir, f"{count}.npy"), features)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

with open(JSON_PATH, "r") as f:
    dataset = json.load(f)

for entry in dataset:
    label = entry["gloss"].lower()

    for inst in entry["instances"]:
        video_id = inst["video_id"]
        bbox = inst["bbox"]
        frame_start = inst["frame_start"]
        frame_end = inst["frame_end"]

        video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            continue

        cap = cv2.VideoCapture(video_path)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
        frame_idx = frame_start

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if frame_end != -1 and frame_idx > frame_end:
                break

            if frame_idx % FRAME_STRIDE != 0:
                frame_idx += 1
                continue

            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = detector.detect(mp_image)
            if result.hand_landmarks:
                features = normalize_landmarks(result.hand_landmarks[0])
                save_sample(label, features)

            frame_idx += 1

        cap.release()

print("\n✅ Dataset built successfully.")