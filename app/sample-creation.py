import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

# ----------------------------
# CONFIGURATION
# ----------------------------
DATA_DIR = "asl_data"         # folder to save samples
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # letters you want to record
SAMPLES_PER_LETTER = 200      # target number of samples per letter

os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------------
# HELPER FUNCTION
# ----------------------------
def normalize_landmarks(hand):
    points = np.array([[lm.x, lm.y, lm.z] for lm in hand])
    origin = points[0]  # wrist
    points -= origin
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0:
        points /= scale
    return points.flatten()  # 63-d vector

def save_sample(letter, features):
    # Create letter folder if it doesn't exist
    letter_dir = os.path.join(DATA_DIR, letter)
    os.makedirs(letter_dir, exist_ok=True)
    # Save with a unique filename
    count = len(os.listdir(letter_dir))
    filename = os.path.join(letter_dir, f"{count}.npy")
    np.save(filename, features)

# ----------------------------
# INITIALIZE MEDIAPIPE
# ----------------------------
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

print("Press a letter key (A-Z) to record that sign.")
print("Press ESC to exit.")

# ----------------------------
# MAIN LOOP
# ----------------------------
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

    cv2.imshow("Hand Tracking - Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

    # Check if key corresponds to a letter
    if chr(key).upper() in LETTERS:
        if result.hand_landmarks:
            features = normalize_landmarks(result.hand_landmarks[0])
            save_sample(chr(key).upper(), features)
            print(f"Saved sample for letter {chr(key).upper()}")
            
cap.release()
cv2.destroyAllWindows()
