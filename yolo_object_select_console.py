import cv2
import torch
import numpy as np
from collections import defaultdict
import threading

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not accessible")

frame_count = 0
previous_objects = {}
selected_tag = None  # e.g. 'bottle_1'

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
            count = sum(1 for k in previous_objects if k.startswith(label))
            new_tag = f"{label}_{count}"
            new_objects[new_tag] = {'class': label, 'center': center, 'bbox': box}

    return new_objects

# 🧵 Threaded input listener
def listen_for_selection():
    global selected_tag
    while True:
        try:
            user_input = input("Select object (class idx): ").strip()
            if not user_input:
                continue
            parts = user_input.split()
            if len(parts) != 2:
                print("Format: class idx (e.g. bottle 1)")
                continue
            class_name, idx = parts[0], parts[1]
            selected_tag = f"{class_name}_{idx}"
            print(f"Selected: {selected_tag}")
        except Exception as e:
            print("Input error:", e)

# Start input thread
threading.Thread(target=listen_for_selection, daemon=True).start()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1

    if frame_count % 10 == 0:
        results = model(frame)
        detections = results.pandas().xyxy[0]
        current_objects = match_objects(detections, previous_objects)
        previous_objects = current_objects

        # print("\nDetected objects:")
        # for tag in current_objects:
        #     print(tag)

    # Draw bounding boxes
    for tag, data in previous_objects.items():
        x1, y1, x2, y2 = data['bbox']
        color = (0, 0, 255) if tag == selected_tag else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, tag, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("YOLOv5 Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
