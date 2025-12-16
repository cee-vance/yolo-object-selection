import cv2
import os
from datetime import datetime

# Set camera indices (adjust if needed)
left_cam_index = 0
right_cam_index = 1

# Open both cameras
cap_left = cv2.VideoCapture(left_cam_index)
cap_right = cv2.VideoCapture(right_cam_index)

# Check if cameras opened successfully
if not cap_left.isOpened() or not cap_right.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

# Create output directory
os.makedirs("stereo_images", exist_ok=True)

print("Press SPACE to capture a stereo pair, ESC to quit.")

while True:
    # Read frames
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        print("Error: Could not read from cameras.")
        break

    # Show both frames side by side
    combined = cv2.hconcat([frame_left, frame_right])
    cv2.imshow("Stereo Cameras (Left | Right)", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break
    elif key == 32:  # SPACE to capture
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        left_filename = f"stereo_images/left_{timestamp}.png"
        right_filename = f"stereo_images/right_{timestamp}.png"
        cv2.imwrite(left_filename, frame_left)
        cv2.imwrite(right_filename, frame_right)
        print(f"Saved pair: {left_filename}, {right_filename}")

# Release resources
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
