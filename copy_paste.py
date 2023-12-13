from ultralytics import YOLO
import cv2
import math
import time
import os
import pyautogui

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 800)  # Adjust width
cap.set(4, 600)  # Adjust height

# Load YOLO model
model = YOLO("best.pt")

# Object classes
classNames = ["closed eye", "open eye"]

# Set the standby time threshold (in seconds)
standby_threshold = 3000  # 5 minutes

# Initialize the last time eyes were open
last_open_time = time.time()

while True:
    success, img = cap.read()
    if not success or img.size == 0:
        continue  # Skip the current iteration if the image is invalid

    results = model(img, stream=True)

    # Coordinates   
    eyes_open = False
    left_eye_closed = False
    right_eye_closed = False

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Class name
            cls = int(box.cls[0])

            # Check if the detected class index is within the range
            if cls < len(classNames):
                if classNames[cls] in ["closed eye", "open eye"]:
                    # Confidence
                    confidence = round(float(box.conf[0]), 2)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to int values

                    # Set color based on class
                    color = (0, 0, 255) if classNames[cls] == "closed eye" else (255, 0, 255)  # Red for closed eyes, Magenta for open eyes
                    # Only consider closed eyes if confidence is above 0.2
                    if confidence > 0.65:
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        

                        # Draw a beautiful label background
                        label_size, base_line = cv2.getTextSize(f"{classNames[cls]}: {confidence}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        y1_label = max(y1 - 10, label_size[1])
                        cv2.rectangle(img, (x1, y1 - label_size[1] - 5), (x1 + label_size[0] + 5, y1), color, cv2.FILLED)
                        # Draw text
                        cv2.putText(img, f"{classNames[cls]}: {confidence}", (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        if classNames[cls] == "closed eye":
                            eyes_open = False
                            # Check if the closed eye is on the left or right side
                            eye_center = (x1 + x2) // 2
                            if eye_center < img.shape[1] // 2:  # Assuming face is centered, check if x-coordinate is on the left side
                                left_eye_closed = True
                            else:
                                right_eye_closed = True

                        elif classNames[cls] == "open eye":
                            eyes_open = True

    # Check which eye is closed
    if left_eye_closed and right_eye_closed:
        print("Both eyes closed")
    elif left_eye_closed:
        pyautogui.hotkey('ctrl', 'c')  # Simulate Ctrl+C
        print("Left eye closed - Simulating Ctrl+C")
    elif right_eye_closed:
        pyautogui.hotkey('ctrl', 'v')  # Simulate Ctrl+V
        print("Right eye closed - Simulating Ctrl+V")

    # Check if eyes are closed for more than the threshold
    if not eyes_open:
        current_time = time.time()
        if current_time - last_open_time > standby_threshold:
            # Put the computer in standby mode
            os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
            last_open_time = current_time

    # Display the webcam feed
    cv2.imshow('Webcam', img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()