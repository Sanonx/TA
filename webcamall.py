# Import necessary modules
from ultralytics import YOLO
import cv2
import json
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # use GPU if available, otherwise use CPU


# Load the YOLO model
model = YOLO("runsMLFLOW/a0facc5975434309b14e1e04d699edf8/artifacts/best.pt")

# Create a dictionary to map class names to category IDs
class_mapping = {
    'person': 0,
    'bicycle': 1,
    'car': 2,
    'motorbike': 3,
    'bus': 4,
    'train': 5,
    'truck': 6,
    'animal': 7,
    'backpack': 8,
    'fire': 9,
    'smoke': 10,
    'flood': 11,
    'doors': 12,
    'stairs': 13,
    'windows': 14,
    'suitcase': 15,
    'cellphone': 16
}

# Create a VideoCapture object for webcam
cap = cv2.VideoCapture(0)  # 0 stands for first webcam, if you have multiple cameras you might want to use 1, 2, etc.

if not cap.isOpened():
    print("Error opening video")

annotations = []

annotation_id = 0

while True:
    # Read a frame from the video
    success, frame = cap.read()

    if not success:
        break

    # Run YOLOv8 inference on the frame
    results = model.predict(frame, stream=True, device=device) 
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy  # assuming the bounding boxes are in (x1, y1, x2, y2) format
            classes = result.boxes.cls
            labels = [result.names[int(c.item())] for c in classes]

            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box.tolist()
                width, height = x2 - x1, y2 - y1
                
                # Create a new annotation_info and append to annotations list
                annotation_info = {
                    "id": annotation_id,
                    "image_id": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                    "category_id": class_mapping[label],
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
                annotations.append(annotation_info)
                annotation_id += 1

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Save the annotations to a json file
    with open('annotations.json', 'w') as f:
        json.dump(annotations, f)

    # Quit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
