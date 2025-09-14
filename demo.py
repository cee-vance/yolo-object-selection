import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m/l/x
results = model(frame)
detections = results.pandas().xyxy[0]  # DataFrame with boxes, class, confidence

# Group and enumerate
from collections import defaultdict

objects = defaultdict(list)
for i, row in detections.iterrows():
    label = row['name']
    objects[label].append((i, row))

# Display
for label, items in objects.items():
    print(f"{label}: {[f'{label}_{i}' for i, _ in enumerate(items)]}")
