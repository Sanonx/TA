import cv2
import pafy
import json
import torch
from ultralytics import YOLO

class YOLOv8Inference:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
    def predict_frame(self, frame):
        return self.model.predict(frame)

class VideoProcessor:
    def __init__(self, model_path, class_mapping, source):
        self.source = source
        self.yolov8_inference = YOLOv8Inference(model_path)
        self.class_mapping = class_mapping

    def process_video(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print("Error opening video")
            return

        annotations = []
        annotation_id = 0

        while True:
            success, frame = cap.read()

            if not success:
                break

            results = self.yolov8_inference.predict_frame(frame)

            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy
                    classes = result.boxes.cls
                    confidences = result.boxes.conf

                    labels = [result.names[int(c.item())] for c in classes]

                    for box, label, confidence in zip(boxes, labels, confidences):
                        x1, y1, x2, y2 = box.tolist()
                        width, height = x2 - x1, y2 - y1

                        annotation_info = {
                            "id": annotation_id,
                            "category_id": self.class_mapping[label],
                            "bbox": [x1, y1, width, height],
                            "area": width * height,
                            "iscrowd": 0,
                            "confidence": float(confidence),
                        }
                        annotations.append(annotation_info)
                        annotation_id += 1

        cap.release()

        with open('annotations.json', 'w') as f:
            json.dump(annotations, f)

if __name__ == "__main__":
    model_path = 'runsMLFLOW/train67/weights/best.pt'
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
    source = 'prueba.mp4'

    video_processor = VideoProcessor(model_path, class_mapping, source)
    video_processor.process_video()
