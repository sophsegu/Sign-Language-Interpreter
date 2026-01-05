import numpy as np
import cv2

#Global variables
background = None
hand = None
frames_elapsed = 0
FRAME_HEIGHT = 500
FRAME_WIDTH = 750
#To edit if not able to recognize and (affects skintone recongition)
CALIBRATION_TIME = 30
BG_WEIGHT = 0.5
OBJ_THRESHOLD = 18

def main():
    region_top = 0
    region_bottom = int(2 * FRAME_HEIGHT / 3)
    region_left = int(FRAME_WIDTH / 2)
    region_right = FRAME_WIDTH

    frames_elapsed = 0
    capture = cv2.VideoCapture(0)

    while (True):
        # Store the frame from the video capture and resize it to the desired window size.
        ret, frame = capture.read()
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        # Flip the frame over the vertical axis so that it works like a mirror, which is more intuitive to the user.
        frame = cv2.flip(frame, 1)

        cv2.imshow("Camera Input", frame)
        frames_elapsed += 1
        # Check if user wants to exit.
        if (cv2.waitKey(1) & 0xFF == ord('x')):
            break

    # When we exit the loop, we have to stop the capture too.
    capture.release()
    cv2.destroyAllWindows()

main()