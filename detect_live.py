from ultralytics import YOLO
import cv2
import time
import signal
import sys

MODEL_PATH = "yolov8n_ncnn_model"
CONF_THRESHOLD = 0.65
CAM_INDEX = 0
WIDTH = 640
HEIGHT = 480

running = True

def stop_handler(sig, frame):
    global running
    running = False

signal.signal(signal.SIGINT, stop_handler)

print("Loading model...")
model = YOLO(MODEL_PATH)

print("Opening camera...")
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

if not cap.isOpened():
    print("Camera failed")
    sys.exit()

print("Real-time detection running (Press Q to quit)")

while running:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(
        frame,
        imgsz=320,
        conf=CONF_THRESHOLD,
        classes=[0],
        device="cpu",
        verbose=False
    )

    count = 0

    for r in results:
        for box in r.boxes:
            count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"{conf:.2f}",(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

    cv2.putText(frame,f"Humans: {count}",(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    cv2.imshow("Drone Vision", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Stopped")
