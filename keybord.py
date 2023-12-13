import cv2
import numpy as np
from ultralytics import YOLO
from math import hypot
import time
last_detection_time = time.time()
detection_interval = 0.06  # 60 milliseconds

cap = cv2.VideoCapture(0)
cap.set(3, 800)  # Adjust width
cap.set(4, 600)  # Adjust height

# Load YOLO model
model = YOLO("best.pt")
text = ""

# Object classes
classNames = ["closed eye", "open eye"]

board = np.zeros((300, 1400), np.uint8)
board[:] = 255

# Keyboard settings
keyboard = np.zeros((600, 1000, 3), np.uint8)
keys_set_1 = {0: "Q", 1: "W", 2: "E", 3: "R", 4: "T",
              5: "A", 6: "S", 7: "D", 8: "F", 9: "G",
              10: "Z", 11: "X", 12: "C", 13: "V", 14: "B"}

def letter(letter_index, text, letter_light):
    # Keys
    if letter_index == 0:
        x = 0
        y = 0
    elif letter_index == 1:
        x = 200
        y = 0
    elif letter_index == 2:
        x = 400
        y = 0
    elif letter_index == 3:
        x = 600
        y = 0
    elif letter_index == 4:
        x = 800
        y = 0
    elif letter_index == 5:
        x = 0
        y = 200
    elif letter_index == 6:
        x = 200
        y = 200
    elif letter_index == 7:
        x = 400
        y = 200
    elif letter_index == 8:
        x = 600
        y = 200
    elif letter_index == 9:
        x = 800
        y = 200
    elif letter_index == 10:
        x = 0
        y = 400
    elif letter_index == 11:
        x = 200
        y = 400
    elif letter_index == 12:
        x = 400
        y = 400
    elif letter_index == 13:
        x = 600
        y = 400
    elif letter_index == 14:
        x = 800
        y = 400

    width = 200
    height = 200
    th = 3 # thickness
    if letter_light is True:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
    else:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 0, 0), th)

    # Text settings
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 10
    font_th = 4
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x
    text_y = int((height + height_text) / 2) + y
    cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (255, 0, 0), font_th)

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN






frames = 0
letter_index = 0

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

            keyboard[:] = (0, 0, 0)
            frames += 1
            new_frame = np.zeros((500, 500, 3), np.uint8)
            if frames == 10:
                letter_index += 1
                frames = 0
            if letter_index == 15:
                letter_index = 0


            for i in range(15):
                if i == letter_index:
                    light = True
                    print(keys_set_1[i]) 
                    current_time = time.time()
                    if current_time - last_detection_time >= detection_interval and classNames[cls] == "closed eye":
                        text=text+keys_set_1[i]
                        last_detection_time = current_time          
                else:
                    light = False
                letter(i, keys_set_1[i], light)

    # Display the webcam feed
    cv2.imshow('Webcam', img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break


    

           

    # Show the text we're writing on the board
    cv2.putText(board, text, (80, 100), font, 9, 0, 3)

    cv2.imshow("Virtual keyboard", keyboard)
    cv2.imshow("Board", board)

    key = cv2.waitKey(100)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()