from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n_ncnn_model")
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

    # ===== DRAW AXES =====
    cv2.line(frame, (w//2, 0), (w//2, h), (255,255,0), 1)
    cv2.line(frame, (0, h//2), (w, h//2), (255,255,0), 1)

    cv2.putText(frame,"X+",(w-40,h//2-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
    cv2.putText(frame,"X-",(10,h//2-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
    cv2.putText(frame,"Y+",(w//2+10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
    cv2.putText(frame,"Y-",(w//2+10,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)

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

        if best_box is not None:
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

            # OPTIONAL: show coordinates
            cv2.putText(frame,f"({cx},{cy})",(cx+10,cy),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)

    cv2.circle(frame, (cx_frame, cy_frame), 5, (255,0,0), -1)

    cv2.putText(frame, f"STATE: {state}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Phase 1 + 2 - Detection & Logic", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
