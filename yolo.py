from ultralytics import YOLO
import cv2
import math

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 800)  # Adjust width
cap.set(4, 600)  # Adjust height

# Load YOLO model
model = YOLO("best.pt")

# Object classes
classNames = ["closed eye", "open eye"]

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # Coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Class name
            cls = int(box.cls[0])

            # Check if the detected class index is within the range
            if cls < len(classNames):
                if classNames[cls] in ["closed eye", "open eye"]:
                    # Confidence
                    confidence = round(float(box.conf[0]), 2)  # Convert tensor to float before rounding

                    # Bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to int values

                    # Set color based on class
                    color = (0, 0, 255) if classNames[cls] == "closed eye" else (255, 0, 255)  # Red for closed eyes, Magenta for open eyes

                    # Only display the result if confidence is above 0.2
                    if confidence > 0.2:
                        # Draw bounding box
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                        # Draw a beautiful label background
                        label_size, base_line = cv2.getTextSize(f"{classNames[cls]}: {confidence}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        y1_label = max(y1 - 10, label_size[1])
                        cv2.rectangle(img, (x1, y1 - label_size[1] - 5), (x1 + label_size[0] + 5, y1), color, cv2.FILLED)

                        # Draw text
                        cv2.putText(img, f"{classNames[cls]}: {confidence}", (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            else:
                print("Unknown class index:", cls)

    # Display the webcam feed
    cv2.imshow('Webcam', img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()