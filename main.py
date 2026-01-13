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
CALIBRATION_TIME = 60
BG_WEIGHT = 0.04
OBJ_THRESHOLD = 15
region_top    = 0
region_bottom = int(2 * FRAME_HEIGHT / 3)
region_left   = int(FRAME_WIDTH / 2)
region_right  = FRAME_WIDTH
MIN_HAND_AREA = 5000

# Here we take the current frame, the number of frames elapsed, and how many fingers we've detected
# so we can print on the screen which gesture is happening (or if the camera is calibrating).
def write_on_image(frame, hand, frames_elapsed):
    hand.update(region_top, region_bottom, region_left, region_right)
    text = ""
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

def get_region(frame):
    # Separate the region of interest from the rest of the frame.
    region = frame[region_top:region_bottom, region_left:region_right]
    # Make it grayscale so we can detect the edges more easily.
    region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    # Use a Gaussian blur to prevent frame noise from being labeled as an edge.
    #region = cv2.GaussianBlur(region, (3,3), 0)

    return region

def get_average(region):
    # We have to use the global keyword because we want to edit the global variable.
    global background
    # If we haven't captured the background yet, make the current region the background.
    if background is None:
        background = region.copy().astype("float")
        return
    # Otherwise, add this captured frame to the average of the backgrounds.
    cv2.accumulateWeighted(region, background, BG_WEIGHT)


def count_fingers(thresholded_image):
    height, width = thresholded_image.shape

    # Create an empty mask
    line_mask = np.zeros_like(thresholded_image)

    # Line at 70% of hand height
    line_height = int(height * 0.7)

    # Draw the line ON THE MASK
    cv2.line(
        line_mask,
        (0, line_height),
        (width, line_height),
        255,
        1
    )

    # AND mask with hand
    line_intersection = cv2.bitwise_and(
        thresholded_image,
        thresholded_image,
        mask=line_mask
    )

    # Find contours on the line
    contours, _ = cv2.findContours(
        line_intersection,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    fingers = 0
    palm_width = abs(hand.right[0] - hand.left[0])

    for cnt in contours:
        if 5 < len(cnt) < palm_width * 0.75:
            fingers += 1

    return fingers



def segment(region):
    global hand

    diff = cv2.absdiff(background.astype(np.uint8), region)
    _, thresholded = cv2.threshold(diff, OBJ_THRESHOLD, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresholded.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        hand.isInFrame = False
        hand.isWaving = False
        hand.waveCounter = 0
        return None

    largest = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest) < MIN_HAND_AREA:
        hand.isInFrame = False
        hand.isWaving = False
        hand.waveCounter = 0
        return None

    hand.isInFrame = True
    return thresholded, largest

def get_hand_data(thresholded_image, segmented_image):
    # Enclose the area around the extremities in a convex hull to connect all outcroppings.
    convexHull = cv2.convexHull(segmented_image)
    
    # Find the extremities for the convex hull and store them as points.
    top    = tuple(convexHull[convexHull[:, :, 1].argmin()][0])
    bottom = tuple(convexHull[convexHull[:, :, 1].argmax()][0])
    left   = tuple(convexHull[convexHull[:, :, 0].argmin()][0])
    right  = tuple(convexHull[convexHull[:, :, 0].argmax()][0])
    
    # Get the center of the palm, so we can check for waving and find the fingers.
    centerX = int((left[0] + right[0]) / 2)
    global hand
    if hand == None:
        hand = HandData(top, bottom, left, right, centerX)
    else:
        hand.update(top, bottom, left, right)
    
    # Only check for waving every 6 frames.
    if frames_elapsed % 6 == 0:
        hand.check_for_waving(centerX)

    hand.gestureList.append(count_fingers(thresholded_image.copy()))
    if frames_elapsed % 12 == 0:
        hand.fingers = most_frequent(hand.gestureList)
        hand.gestureList.clear()
    

def most_frequent(input_list):
    dict = {}
    count = 0
    most_freq = 0
    
    for item in reversed(input_list):
        dict[item] = dict.get(item, 0) + 1
        if dict[item] >= count :
            count, most_freq = dict[item], item
    
    return most_freq

def main():
    frames_elapsed = 0
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while (True):
        # Store the frame from the video capture and resize it to the desired window size.
        ret, frame = capture.read()
        if not ret or frame is None:
            print("Frame not captured")
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        # Flip the frame over the vertical axis so that it works like a mirror, which is more intuitive to the user.
        frame = cv2.flip(frame, 1)

        # Separate the region of interest and prep it for edge detection.
        region = get_region(frame)
        
        if frames_elapsed < CALIBRATION_TIME:
            get_average(region)
        else:
            region_pair = segment(region)
            if region_pair is not None:
                # If we have the regions segmented successfully, show them in another window for the user.
                (thresholded_region, segmented_region) = region_pair
                get_hand_data(thresholded_region, segmented_region)

                cv2.drawContours(region, [segmented_region], -1, (255, 255, 255))
                cv2.imshow("Segmented Image", region)
        # Write the action the hand is doing on the screen, and draw the region of interest.
        write_on_image(frame, hand, frames_elapsed)
        cv2.imshow("Camera Input", frame)
        frames_elapsed += 1
        # Check if user wants to exit.
        if (cv2.waitKey(1) & 0xFF == ord('x')):
            break

        if (cv2.waitKey(1) & 0xFF == ord('r')):
            frames_elapsed = 0

    # When we exit the loop, we have to stop the capture too.
    capture.release()
    cv2.destroyAllWindows()

main()