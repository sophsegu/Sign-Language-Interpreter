import numpy as np
import cv2
import HandData

#Global variables
background = None
hand = HandData.HandData(0,0,0,0,0)
frames_elapsed = 0
FRAME_HEIGHT = 500
FRAME_WIDTH = 750
#To edit if not able to recognize and (affects skintone recongition)
CALIBRATION_TIME = 30
BG_WEIGHT = 0.5
OBJ_THRESHOLD = 18

# Here we take the current frame, the number of frames elapsed, and how many fingers we've detected
# so we can print on the screen which gesture is happening (or if the camera is calibrating).
def write_on_image(frame, hand, frames_elapsed):
    region_top    = 0
    region_bottom = int(2 * FRAME_HEIGHT / 3)
    region_left   = int(FRAME_WIDTH / 2)
    region_right  = FRAME_WIDTH

    hand.update(region_top, region_bottom, region_left, region_right)
    if frames_elapsed < CALIBRATION_TIME:
        text = "Calibrating..."
    elif hand == None or not hand.isInFrame:
        text = "No hand detected"
    else:
        if hand.isWaving:
            text = "Waving"
        elif hand.fingers == 0:
            text = "Rock"
        elif hand.fingers == 1:
            text = "Pointing"
        elif hand.fingers == 2:
            text = "Scissors"
    
    cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.4,( 0 , 0 , 0 ),2,cv2.LINE_AA)
    cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.4,(255,255,255),1,cv2.LINE_AA)

    # Highlight the region of interest using a drawn rectangle.
    cv2.rectangle(frame, (region_left, region_top), (region_right, region_bottom), (255,255,255), 2)

def main():
    frames_elapsed = 0
    capture = cv2.VideoCapture(0)

    while (True):
        # Store the frame from the video capture and resize it to the desired window size.
        ret, frame = capture.read()
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        # Flip the frame over the vertical axis so that it works like a mirror, which is more intuitive to the user.
        frame = cv2.flip(frame, 1)
        
        # Write the action the hand is doing on the screen, and draw the region of interest.
        write_on_image(frame, hand, frames_elapsed)
        cv2.imshow("Camera Input", frame)
        frames_elapsed += 1
        # Check if user wants to exit.
        if (cv2.waitKey(1) & 0xFF == ord('x')):
            break

    # When we exit the loop, we have to stop the capture too.
    capture.release()
    cv2.destroyAllWindows()

main()