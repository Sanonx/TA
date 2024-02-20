"""
This script performs real-time object detection using the YOLOv8 model and generates annotation information in COCO format for each detected object. The detected objects are visualized on the input video stream in real-time, and their annotations are saved to a JSON file.

Args:
    model_path (str): Path to the YOLOv8 model file (e.g., 'model.pt').
    source (str): Source of the input video stream (e.g., 'video.mp4' or a YouTube URL).
    output_json (str): Path to the output JSON file for annotations.
"""

import cv2
import json
import torch
from ultralytics import YOLO

class ObjectDetection:
    """
    A class for real-time object detection using YOLOv8 model.

    Args:
        model_path (str): Path to the YOLOv8 model file.
    """
    def __init__(self, model_path):
        # Check and set device (GPU o CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Path to YOLO model
        self.model = YOLO(model_path)

        # Create a dictionary to map class names to category IDs
        self.class_mapping = {
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

        self.class_names = list(self.class_mapping.keys())  # Lista de nombres de clases

    def detect_objects(self, frame):
        """
        Detect objects in a frame.

        Args:
            frame (numpy.ndarray): Input frame for object detection.

        Returns:
            list: List of dictionaries containing object detection results.
        """        
        results = self.model.predict(frame, stream=True, device=self.device)
        annotations = []

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy  # assuming the bounding boxes are in (x1, y1, x2, y2) format (YOLO format)
                classes = result.boxes.cls
                confidences = result.boxes.conf  # Get confidence scores

                labels = [self.class_names[int(c.item())] for c in classes]  # Usar los nombres de las clases

                for box, label, confidence in zip(boxes, labels, confidences):
                    x1, y1, x2, y2 = box.tolist()
                    width, height = x2 - x1, y2 - y1

                    # Create a new annotation_info and append to annotations list
                    annotation_info = {
                        "category_id": label,
                        "bbox": [x1, y1, width, height],
                        "confidence": float(confidence),  # Add confidence to the annotation
                    }
                    annotations.append(annotation_info)

        return annotations

class JSONExporter:
    """
    A class for exporting annotations to a JSON file.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.annotations = []

    def add_annotations(self, frame_number, annotations):
        for annotation in annotations:
            annotation_info = {
                "image_id": frame_number,
                "category_id": annotation["category_id"],
                "bbox": annotation["bbox"],
                "confidence": annotation["confidence"],
            }
            self.annotations.append(annotation_info)

    def export_json(self):
        with open(self.file_path, 'w') as f:
            json.dump(self.annotations, f)

class ImageVisualizer:
    """
    A class for real-time visualization of object detection results.
    """
    def __init__(self, window_name="Object Detection"):
        self.window_name = window_name

    def visualize_image(self, frame, annotations):
        for annotation in annotations:
            x, y, w, h = annotation["bbox"]
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            class_name = annotation["category_id"]
            confidence = annotation["confidence"]

            # Create combined text with label and confidence
            label = f"{class_name}: {confidence:.2f}"

            # Get the size of the combined text
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # Adjust label position
            if x2 + text_size[0] < frame.shape[1]:  # Check if the combined text fits inside the frame
                x_text = x2
            else:
                x_text = frame.shape[1] - text_size[0] - 5

            # Adjust vertical position
            y_text = max(y1 - 10, text_size[1] + 10)

            # Draw the bounding box and the combined text
            color = (0, 255, 0)  # Green color
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

if __name__ == "__main__":
    model_path = "runsMLFLOW/a0facc5975434309b14e1e04d699edf8/artifacts/best.pt"
    object_detector = ObjectDetection(model_path)
    json_exporter = JSONExporter('annotations.json')
    image_visualizer = ImageVisualizer()

    source = 'VideosBursa/GH010079.MP4'
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error opening video")

    frame_number = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        detections = object_detector.detect_objects(frame)
        json_exporter.add_annotations(frame_number, detections)
        json_exporter.export_json()

        image_visualizer.visualize_image(frame, detections)

        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()







