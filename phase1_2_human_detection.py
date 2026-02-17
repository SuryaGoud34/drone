from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

CENTER_TOL = 40
AREA_DESCEND_TH = 0.08
CONF_TH = 0.4

state = "SEARCH"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_area = h * w
    cx_frame, cy_frame = w // 2, h // 2

    results = model(frame, classes=[0], conf=CONF_TH)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        state = "SEARCH"
    else:
        max_area = 0
        best_box = None

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_box = (x1, y1, x2, y2)

        x1, y1, x2, y2 = best_box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        area_ratio = max_area / frame_area

        if abs(cx - cx_frame) < CENTER_TOL and abs(cy - cy_frame) < CENTER_TOL:
            if area_ratio < AREA_DESCEND_TH:
                state = "DESCEND"
            else:
                state = "HOVER"
        else:
            state = "ALIGN"

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)

    cv2.circle(frame, (cx_frame, cy_frame), 5, (255,0,0), -1)
    cv2.putText(frame, f"STATE: {state}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Phase 1 + 2 - Detection & Logic", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()