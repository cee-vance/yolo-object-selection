import cv2
import torch
import numpy as np
from collections import defaultdict
import threading
import serial
import math

# === Global State ===
last_sent_position = None
selected_tag = None
selection_updated = False
state_lock = threading.Lock()

# === Camera to Servo Mapping ===
def pixel_to_angle(x, y, frame_width=640, frame_height=480, hfov=60, vfov=45):
    dx = x - frame_width / 2
    dy = y - frame_height / 2

    # Invert X-axis to correct servo direction
    angle_x = -(dx / frame_width) * hfov
    angle_y = -(dy / frame_height) * vfov
    return angle_x, angle_y

def angle_to_servo(angle_x, angle_y):
    servo_x = int(90 + angle_x)
    servo_y = int(90 + angle_y)
    return max(0, min(180, servo_x)), max(0, min(180, servo_y))

# === Arduino Serial Setup ===
arduino = serial.Serial('COM5', 9600)

def send_to_arduino(servo_x, servo_y):
    command = f"{servo_x},{servo_y}\n"
    arduino.write(command.encode())



# === YOLOv5 Setup ===
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not accessible")

frame_count = 0
previous_objects = {}

# === Object Matching ===
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

# === Console Input Thread ===
def listen_for_selection_console():
    global selected_tag, selection_updated
    while True:
        try:
            user_input = input("Select object (class index): ").strip()
            if not user_input:
                continue
            parts = user_input.split()
            if len(parts) != 2:
                print("⚠️ Format should be: class index (e.g. bottle 0)")
                continue

            class_name, idx = parts[0], parts[1]
            with state_lock:
                selected_tag = f"{class_name}_{idx}"
                selection_updated = True
            print(f"✅ Selected: {selected_tag}")
        except Exception as e:
            print(f"⚠️ Input error: {e}")

import speech_recognition as sr

def listen_for_selection_voice():
    global selected_tag, selection_updated
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)

    while True:
        try:
            print("🎙️ Say object selection (e.g. 'bottle one')...")
            with mic as source:
                audio = recognizer.listen(source,timeout=10,phrase_time_limit=4)
            command = recognizer.recognize_google(audio).lower().strip()
            print(f"🗣️ Heard: {command}")

            parts = command.split()
            if len(parts) != 2:
                print("⚠️ Format should be: class index (e.g. 'bottle one')")
                continue

            class_name = parts[0]
            idx_word = parts[1]

            # Convert spoken number to digit
            word_to_digit = {
                "zero": "0", "one": "1", "two": "2", "three": "3",
                "four": "4", "five": "5", "six": "6", "seven": "7",
                "eight": "8", "nine": "9"
            }
            idx = word_to_digit.get(idx_word, None)
            if idx is None:
                print("⚠️ Could not interpret index")
                continue

            selected_tag = f"{class_name}_{idx}"
            selection_updated = True
            print(f"✅ Selected: {selected_tag}")

        except sr.UnknownValueError:
            pass
        except sr.RequestError as e:
            pass
        except Exception as e:
            pass

threading.Thread(target=listen_for_selection_console, daemon=True).start()

# === Main Loop ===
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

    # Draw bounding boxes
    with state_lock:
        tag = selected_tag
        updated = selection_updated

    for obj_tag, data in previous_objects.items():
        x1, y1, x2, y2 = data['bbox']
        color = (0, 0, 255) if obj_tag == tag else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, obj_tag, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Handle targeting once per selectionq
    if updated and tag in previous_objects:
        x1, y1, x2, y2 = previous_objects[tag]['bbox']
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2

        angle_x, angle_y = pixel_to_angle(x_center, y_center)
        servo_x, servo_y = angle_to_servo(angle_x, angle_y)

        new_position = (servo_x, servo_y)
        if new_position != last_sent_position:
            print("🎯 Sending Arduino:", new_position)
            send_to_arduino(servo_x, servo_y)
            last_sent_position = new_position

        with state_lock:
            selection_updated = False

    cv2.imshow("YOLOv5 Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
