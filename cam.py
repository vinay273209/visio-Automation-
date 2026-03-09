import cv2
from ultralytics import YOLO

# load YOLO model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

def get_head_counts():

    ret, frame = cap.read()
    if not ret:
        return 0,0,0,0

    height, width, _ = frame.shape

    mid_x = width // 2
    mid_y = height // 2

    zone1 = 0
    zone2 = 0
    zone3 = 0
    zone4 = 0

    results = model(frame)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])

            if cls == 0:   # person

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if cx < mid_x and cy < mid_y:
                    zone1 += 1
                elif cx >= mid_x and cy < mid_y:
                    zone2 += 1
                elif cx < mid_x and cy >= mid_y:
                    zone3 += 1
                else:
                    zone4 += 1

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    # draw zone lines
    cv2.line(frame,(mid_x,0),(mid_x,height),(255,0,0),2)
    cv2.line(frame,(0,mid_y),(width,mid_y),(255,0,0),2)

    cv2.putText(frame,f"Zone1: {zone1}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
    cv2.putText(frame,f"Zone2: {zone2}",(mid_x+10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
    cv2.putText(frame,f"Zone3: {zone3}",(10,mid_y+30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
    cv2.putText(frame,f"Zone4: {zone4}",(mid_x+10,mid_y+30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

    cv2.imshow("4 Frame Head Count System",frame)

    return zone1, zone2, zone3, zone4