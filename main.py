import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Load model ---
base_options = python.BaseOptions(
    model_asset_path=r"C:\Users\sophs\OneDrive\Desktop\Sign-Language-Interpreter\hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)

# --- Open webcam ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# --- Create window BEFORE loop ---
cv2.namedWindow("CamOutput", cv2.WINDOW_NORMAL)

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)

    # --- IMPORTANT FIX ---
    # Convert BGR → RGB and make a COPY
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = frame_rgb.copy()

    h, w, _ = frame.shape

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    result = detector.detect(mp_image)

    # Draw landmarks on ORIGINAL BGR frame
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            for lm in hand_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 6, (255, 0, 255), cv2.FILLED)

    cv2.imshow("CamOutput", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
