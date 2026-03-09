import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from collections import deque
import math

# -------------------------
# Load Models
# -------------------------
yolo = YOLO("yolov8s.pt")

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose()
hands = mp_hands.Hands()
face = mp_face.FaceDetection()

# store last 10 wrist positions
movement_history = deque(maxlen=10)

cap = cv2.VideoCapture(0)

def dist(a,b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    activity = "None"

    # -------------------------
    # Object detection
    # -------------------------
    results = yolo(frame, conf=0.3)

    objects = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = yolo.names[cls]
            objects.append(name)

    annotated = results[0].plot()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # -------------------------
    # Pose detection
    # -------------------------
    pose_res = pose.process(rgb)

    # -------------------------
    # Face detection
    # -------------------------
    face_res = face.process(rgb)

    # -------------------------
    # Hand detection
    # -------------------------
    hand_res = hands.process(rgb)

    if pose_res.pose_landmarks:

        mp_draw.draw_landmarks(
            annotated,
            pose_res.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        lm = pose_res.pose_landmarks.landmark

        nose = lm[mp_pose.PoseLandmark.NOSE]
        wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

        movement_history.append((wrist.x, wrist.y))

        # -------------------------
        # Temporal motion analysis
        # -------------------------
        movement = 0

        if len(movement_history) > 1:

            for i in range(len(movement_history)-1):

                x1,y1 = movement_history[i]
                x2,y2 = movement_history[i+1]

                movement += abs(x2-x1) + abs(y2-y1)

        # -------------------------
        # Drinking detection
        # -------------------------
        if "cup" in objects or "bottle" in objects:

            if dist(wrist,nose) < 0.1:
                activity = "Drinking"

        # -------------------------
        # Eating detection
        # -------------------------
        if ("bowl" in objects or "spoon" in objects):

            if movement > 0.05:
                activity = "Eating"

        # -------------------------
        # Mobile usage
        # -------------------------
        if "cell phone" in objects:
            activity = "Using Mobile"

        # -------------------------
        # Gaming detection
        # -------------------------
        if "keyboard" in objects or "laptop" in objects:
            activity = "Playing Game"

    # -------------------------
    # Display activity
    # -------------------------
    cv2.putText(
        annotated,
        activity,
        (40,60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        3
    )

    cv2.imshow("AI Activity Recognition", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()