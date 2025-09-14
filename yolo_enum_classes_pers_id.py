import cv2
import torch
import numpy as np
from collections import defaultdict

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not accessible")

frame_count = 0
previous_objects = {}  # Persistent object tags

def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def match_objects(new_detections, previous_objects, threshold=50):
    new_objects = {}
    used_tags = set()

    for row in new_detections.itertuples():
        label = row.name
        box = [int(row.xmin), int(row.ymin), int(row.xmax), int(row.ymax)]
        center = get_center(box)

        # Try to match with previous objects of the same class
        best_match = None
        best_dist = float('inf')
        for tag, data in previous_objects.items():
            if data['class'] != label or tag in used_tags:
                continue
            dist = np.linalg.norm(np.array(center) - np.array(data['center']))
            if dist < best_dist:
                best_dist = dist
                best_match = tag

        if best_match and best_dist < threshold:
            new_objects[best_match] = {'class': label, 'center': center, 'bbox': box}
            used_tags.add(best_match)
        else:
            # Assign new index
            count = sum(1 for k in previous_objects if k.startswith(label))
            new_tag = f"{label}_{count}"
            new_objects[new_tag] = {'class': label, 'center': center, 'bbox': box}

    return new_objects

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1

    # Run detection every 10th frame
    if frame_count % 10 == 0:
        results = model(frame)
        detections = results.pandas().xyxy[0]
        current_objects = match_objects(detections, previous_objects)
        previous_objects = current_objects

        # Print enumerated tags
        print("\nDetected objects:")
        for tag in current_objects:
            print(tag)

    # Draw bounding boxes from previous_objects
    for tag, data in previous_objects.items():
        x1, y1, x2, y2 = data['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, tag, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("YOLOv5 Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
