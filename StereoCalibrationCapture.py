import cv2
import os
from datetime import datetime

# Use a single camera index (adjust if needed)
cam_index = 1
cap = cv2.VideoCapture(cam_index)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Create output directory
os.makedirs("stereo_images", exist_ok=True)

print("Press SPACE to capture a stereo pair, ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from camera.")
        break

    # Split the stitched frame into left and right halves
    height, width = frame.shape[:2]
    mid = width // 2
    frame_left = frame[:, :mid]
    frame_right = frame[:, mid:]

    # Show both halves side by side (optional, just to visualize)
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
cap.release()
cv2.destroyAllWindows()
