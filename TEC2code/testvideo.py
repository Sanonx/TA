from ultralytics import YOLO
import cv2
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import numpy as np
import pafy
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # use GPU if available, otherwise use CPU

# Cargar el modelo
# model = YOLO("runsMLFLOW/train41/weights/best.pt")
model = YOLO("yolov8n.pt")
# Define source as YouTube video URL
source = 'prueba.mp4'

# En caso de que la resolucion de entrada no sea 640x480 que viene de Thales, el rescalado en la inferencia se hara automaticamente conforme a la arquitectura de YOLO

best = source


# videoPafy = pafy.new(source)
# best = videoPafy.getbest(preftype="mp4")

# Create a VideoCapture object
# cap = cv2.VideoCapture(best.url)
cap = cv2.VideoCapture(best)

if not cap.isOpened():
    print("Error opening video")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    
    # Run YOLO detection on the frame
    results = model.predict(frame, stream=True, device=device, imgsz= [640, 480]) 

    # Extract each detection and render it on the frame
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy  # assuming the bounding boxes are in (x1, y1, x2, y2) format
            classes = result.boxes.cls
            conf = result.boxes.conf
            labels = [result.names[int(c.item())] for c in classes]

            for box, label, score in zip(boxes, labels, conf):
                x1, y1, x2, y2 = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())
                label = f"{label}: {score.item():.2f}"  # add confidence score to label
                combined_text = label + " " + f"Conf: {score.item():.2f}"
                
                # Get the size of the combined text
                text_size, _ = cv2.getTextSize(combined_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

                # Adjust label position
                if x2 + text_size[0] < frame.shape[1]:  # Check if the combined text fits inside the frame
                    x_text = x2
                else:
                    x_text = frame.shape[1] - text_size[0] - 5

                # Adjust vertical position
                y_text = max(y1 - 10, text_size[1] + 10)

                # Draw the bounding box and the combined text
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, combined_text, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLOv8 Inference', frame)

    # Quit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the video after looping through all frames
cap.release()
cv2.destroyAllWindows() 

# CLI: yolo task=detect mode=predict model=yolov8x.pt device=cpu source=0 show=True task=detect mode=predict model=TeamAware/runsMLFLOW/train27/weights/best.pt conf=0.25 source=
# https://www.youtube.com/watch?v=Fb-A88ho0sA
# https://www.youtube.com/watch?v=lZoZiqUFGlA
# https://www.youtube.com/watch?v=Cd43nNGvh00
# https://www.youtube.com/watch?v=w_qZ7vHabWI
# https://www.youtube.com/watch?v=VcywkVphqgY