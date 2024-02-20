import cv2
import json
import numpy as np
import time

# Create a dictionary to map class names to category IDs
class_mapping = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorbike',
    4: 'bus',
    5: 'train',
    6: 'truck',
    7: 'animal',
    8: 'backpack',
    9: 'fire',
    10: 'smoke',
    11: 'flood',
    12: 'doors',
    13: 'stairs',
    14: 'windows',
    15: 'suitcase',
    16: 'cellphone'
}

# Create a VideoCapture object for webcam
cap = cv2.VideoCapture(0)  # 0 stands for first webcam

# Load the annotations
with open("annotations.json", "r") as f:
    annotations = json.load(f)

# Loop through the video frames
frame_counter = 0
while cap.isOpened():
    time.sleep(0.1)
    success, frame = cap.read()

    if not success:
        break

    # Get the current annotations
    current_annotations = [a for a in annotations if a["image_id"] == frame_counter]

    # Draw the annotations on the frame
    for annotation in current_annotations:
        bbox = annotation.get("bbox", [])
        if bbox:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Get the class label
            class_id = annotation.get("category_id", 0)
            class_label = class_mapping.get(class_id, "Unknown")
            cv2.putText(frame, class_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Quit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_counter += 1

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
