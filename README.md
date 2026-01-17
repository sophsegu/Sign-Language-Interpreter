# Sign Language Interpreter (Computer Vision + AI)

This project is an **AI-powered sign language recognition system** that uses a webcam to detect hands, extract landmarks, interpret finger positions, and (eventually) translate signs into text.

The system is built using **MediaPipe Tasks API**, **OpenCV**, and Python. It is designed to be extended into a full sign-to-text pipeline.

---

## What This Project Does (Current State)

- Opens your webcam
- Detects hands in real time
- Extracts **21 hand landmarks** per hand
- Prints normalized **(x, y, z)** coordinates for each landmark
- Enables logic to determine which fingers are up/down

Planned:

* Gesture classification (e.g. 👍 thumbs up)
* ASL letter recognition
* Dynamic gesture recognition (movement over time)
* Natural language transcription using an LLM

---

## Core Technologies

* **Python 3.13**
* **MediaPipe Tasks API** (NOT `mp.solutions`)
* **OpenCV** (camera + visualization)
* **TensorFlow Lite** (used internally by MediaPipe)


---

##  Project Structure

```
Sign-Language-Interpreter/
│
├── main.py
├── hand_landmarker.task
├── README.md
```

---

##  Installation

###  Download the hand model

You **must** download this file:

```
hand_landmarker.task
```

Place it in the **root of the project folder**.

---

##  How to Run

From the project folder:

```bash
python main.py
```

Controls:

* Press **ESC** to exit

---


##  Hand Landmarks Explained

MediaPipe detects **21 landmarks per hand**:

```
0  → Wrist
1–4  → Thumb
5–8  → Index finger
9–12 → Middle finger
13–16 → Ring finger
17–20 → Pinky
```

Each landmark has:

```
x → horizontal position (0 = left, 1 = right)
y → vertical position   (0 = top, 1 = bottom)
z → depth (negative = closer to camera)
```

Example output:

```
8: x=0.325, y=0.069, z=-0.094
```

Means:

* Index fingertip
* Near top of image
* Slightly closer to camera

##  Common Errors & Fixes

###  Black or noisy camera feed

* Ensure no other app is using the camera
* Try changing:

```python
cv2.VideoCapture(0)
```

to:

```python
cv2.VideoCapture(1)
```

---

