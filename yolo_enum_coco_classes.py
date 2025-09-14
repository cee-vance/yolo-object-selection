import cv2
import torch
from collections import defaultdict

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not accessible")

frame_count = 0
last_detections = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1

    # Run detection every 10th frame
    if frame_count % 10 == 0:
        results = model(frame)
        last_detections = results.pandas().xyxy[0]

        # Enumerate and print detected classes
        class_map = defaultdict(list)
        for i, row in last_detections.iterrows():
            label = row['name']
            class_map[label].append((i, row))

        print("\nDetected objects:")
        for label, items in class_map.items():
            print(f"{label}: {[f'{label}_{idx}' for idx, _ in enumerate(items)]}")

    # Draw bounding boxes from last detection
    if last_detections is not None:
        class_map = defaultdict(list)
        for i, row in last_detections.iterrows():
            label = row['name']
            class_map[label].append((i, row))

        for label, items in class_map.items():
            for idx, (i, row) in enumerate(items):
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                tag = f"{label}_{idx}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, tag, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("YOLOv5 Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
