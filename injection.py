"""
This script performs object detection using YOLOv8 model and generates annotation information in COCO format for each detected object.

Req: 
model: .pt file downloaded local, trained model or default model.
source: mp4 file, youtube link or any other specified in YOLO predict inference sources documentation
"""

# Import necessary modules
from ultralytics import YOLO
import cv2
import pafy # Pafy only necessary if u wan't to do a inference on youtube url.
import json
import torch 

# Check and set device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Path to YOLO model, change  to your local model path
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

# Define source (video file or YouTube URL)
source = 'prueba.mp4'
# source = 'rtsp://54.220.127.65:8554/drone' # Example of rtsp source

if source.startswith("https://www.youtube.com"):
    # Si no termina con ".mp4", asumimos que es una URL y tratamos de obtener el mejor enlace de descarga mp4
    videoPafy = pafy.new(source)
    best = videoPafy.getbest(preftype="mp4")
    source = best.url

cap = cv2.VideoCapture(source)

# Check if video source is opened successfully
if not cap.isOpened():
    print("Error opening video")

annotations = []
annotation_id = 0

# Loop through video frames
while True:
    # Read a frame from the video
    success, frame = cap.read()

    if not success:
        break

    # Run YOLOv8 inference on the frame
    results = model.predict(frame, stream=True, device=device) 
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy  # assuming the bounding boxes are in (x1, y1, x2, y2) format (YOLO format)
            classes = result.boxes.cls
            confidences = result.boxes.conf  # Get confidence scores

            labels = [result.names[int(c.item())] for c in classes]

            for box, label, confidence in zip(boxes, labels, confidences):
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
                    "confidence": float(confidence),  # Add confidence to the annotation
                }
                annotations.append(annotation_info)
                annotation_id += 1

    # Save the annotations to a json file
    with open('annotations.json', 'w') as f:
        json.dump(annotations, f)

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

