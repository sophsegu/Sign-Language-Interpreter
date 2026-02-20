import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, TimeDistributed, BatchNormalization

# ---------------------------
# Paths and constants
# ---------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_JSON = os.path.join(PROJECT_ROOT, "archive", "WLASL_v0.3.json")
VIDEOS_DIR = os.path.join(PROJECT_ROOT, "archive", "videos")
CNN_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "cnn_lstm_model.h5")
LABELS_PATH = os.path.join(PROJECT_ROOT, "models", "labels.npy")

SEQ_LENGTH = 16
FRAME_SIZE = (64, 64)

# ---------------------------
# Step 1: Parse JSON and collect video metadata
# ---------------------------
with open(DATA_JSON, "r") as f:
    data = json.load(f)

video_instances = []
for item in data:
    gloss = item["gloss"]
    for instance in item["instances"]:
        video_path = os.path.join(VIDEOS_DIR, f"{instance['video_id']}.mp4")
        if os.path.exists(video_path):
            video_instances.append({
                "video_path": video_path,
                "gloss": gloss,
                "bbox": instance.get("bbox", None),
                "split": instance.get("split", "train")
            })

print(f"Total videos found: {len(video_instances)}")

# ---------------------------
# Step 2: Function to extract frames
# ---------------------------
import cv2

def extract_frames(video_path, bbox=None, seq_length=SEQ_LENGTH):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if bbox:
            x1, y1, x2, y2 = bbox
            frame = frame[y1:y2, x1:x2]
        frame = cv2.resize(frame, FRAME_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) < seq_length:
        while len(frames) < seq_length:
            frames.append(frames[-1])
    else:
        idxs = np.linspace(0, len(frames)-1, seq_length, dtype=int)
        frames = [frames[i] for i in idxs]

    return np.array(frames, dtype=np.float32) / 255.0

# ---------------------------
# Step 3: Preprocess all videos
# ---------------------------
sequences = []
labels = []

for inst in tqdm(video_instances, desc="Processing videos"):
    seq = extract_frames(inst["video_path"], bbox=inst["bbox"])
    sequences.append(seq)
    labels.append(inst["gloss"])

sequences = np.array(sequences)
labels = np.array(labels)
print("Sequences shape:", sequences.shape)

# ---------------------------
# Step 4: Encode labels
# ---------------------------
le = LabelEncoder()
y_int = le.fit_transform(labels)
y_onehot = tf.keras.utils.to_categorical(y_int)
num_classes = len(le.classes_)
np.save(LABELS_PATH, le.classes_)
print(f"Number of classes: {num_classes}")

# ---------------------------
# Step 5: Train / Validation split
# ---------------------------
X_train, X_val, y_train, y_val = train_test_split(
    sequences, y_onehot, test_size=0.1, random_state=42, stratify=y_onehot
)

# ---------------------------
# Step 6: Build CNN+LSTM model
# ---------------------------
SEQ_LENGTH = X_train.shape[1]
FRAME_HEIGHT, FRAME_WIDTH = X_train.shape[2:4]

model = Sequential([
    TimeDistributed(Conv2D(32, (3,3), activation='relu'), input_shape=(SEQ_LENGTH, FRAME_HEIGHT, FRAME_WIDTH,3)),
    TimeDistributed(MaxPooling2D((2,2))),
    TimeDistributed(BatchNormalization()),

    TimeDistributed(Conv2D(64, (3,3), activation='relu')),
    TimeDistributed(MaxPooling2D((2,2))),
    TimeDistributed(BatchNormalization()),

    TimeDistributed(Flatten()),

    LSTM(128, return_sequences=False),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---------------------------
# Step 7: Train the model
# ---------------------------
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=8
)

# ---------------------------
# Step 8: Save model
# ---------------------------
model.save(CNN_MODEL_PATH)
print(f"Model saved at {CNN_MODEL_PATH}")
