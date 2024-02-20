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
            'cellphone': 16,
            'rails': 17
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
