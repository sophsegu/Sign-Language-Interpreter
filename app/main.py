import cv2
import numpy as np
import collections
import tensorflow as tf
import os

# ---------------------------
# Paths
# ---------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CNN_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "cnn_lstm_model.h5")
LABELS_PATH = os.path.join(PROJECT_ROOT, "models", "labels.npy")

# ---------------------------
# Load model and labels
# ---------------------------
model = tf.keras.models.load_model(CNN_MODEL_PATH)
labels = np.load(LABELS_PATH)
SEQ_LENGTH = 16
FRAME_SIZE = (64, 64)
frame_buffer = collections.deque(maxlen=SEQ_LENGTH)

# ---------------------------
# Function to display subtitles
# ---------------------------
def draw_subtitle(frame, text, alpha=0.5):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (255, 255, 255)
    bg_color = (255, 255, 255)

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    x = (w - text_width) // 2
    y = h - 20

    overlay = frame.copy()
    cv2.rectangle(overlay, (x-10, y-text_height-10), (x+text_width+10, y+baseline+10), bg_color, cv2.FILLED)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, font_thickness)

# ---------------------------
# Open webcam
# ---------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_resized = cv2.resize(frame, FRAME_SIZE)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_buffer.append(frame_rgb / 255.0)

    predicted_text = ""
    if len(frame_buffer) == SEQ_LENGTH:
        X_input = np.expand_dims(np.array(frame_buffer), axis=0)
        pred = model.predict(X_input, verbose=0)
        predicted_class = np.argmax(pred)
        predicted_text = labels[predicted_class]

    if predicted_text:
        draw_subtitle(frame, predicted_text, alpha=0.5)

    cv2.imshow("Sign Language Interpreter", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
