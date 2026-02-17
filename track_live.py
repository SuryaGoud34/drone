import cv2
import numpy as np
from ultralytics import YOLO
from sort_tracker import Tracker

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

trackers = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)[0]

    detections = []

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls != 0:
            continue

        x1,y1,x2,y2 = map(int, box.xyxy[0])
        detections.append(np.array([x1,y1,x2,y2]))

    # update trackers
    updated = []
    for det in detections:
        matched=False
        for trk in trackers:
            if trk.time_since_update<5:
                i = np.sum((trk.predict()-det)**2)
                if i<2000:
                    trk.update(det)
                    updated.append(trk)
                    matched=True
                    break
        if not matched:
            updated.append(Tracker(det))

    trackers = updated

    # draw
    nearest=None
    nearest_dist=999999

    for trk in trackers:
        box = trk.predict().astype(int)
        x1,y1,x2,y2 = box

        cx=(x1+x2)//2
        cy=(y1+y2)//2
        dist = frame.shape[0]-cy

        if dist<nearest_dist:
            nearest_dist=dist
            nearest=trk.id

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,f"ID {trk.id}",(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    if nearest is not None:
        cv2.putText(frame,f"TARGET ID: {nearest}",
                    (20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

    cv2.imshow("Tracking",frame)
    if cv2.waitKey(1)==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
