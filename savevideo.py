from ultralytics import YOLO
import cv2
import numpy as np
import pafy
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar el modelo
model = YOLO("runsMLFLOW/train67/weights/best.pt")

source = 'VideosBursa/GH010078.MP4'


if source.startswith("https://www.youtube.com"):
    # Si no termina con ".mp4", asumimos que es una URL y tratamos de obtener el mejor enlace de descarga mp4
    videoPafy = pafy.new(source)
    best = videoPafy.getbest(preftype="mp4")
    source = best.url

cap = cv2.VideoCapture(source)

# Definir el nombre del archivo de salida
output_filename = 'videoBursa78.mp4'


if not cap.isOpened():
    print("Error opening video")

# Obtener la información del video de entrada
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


# Configurar el video de salida
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, 30, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO detection on the frame
    results = model.predict(frame, stream=True, device=device, imgsz=[640, 480])

    # Extract each detection and render it on the frame
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy
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
                cv2.putText(frame, combined_text, (x_text, y_text), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame with annotations to the output video
    out.write(frame)

# Release video objects
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video guardado como {output_filename}")

# CLI: yolo task=detect mode=predict model=yolov8x.pt device=cpu source=0 show=True task=detect mode=predict model=TeamAware/runsMLFLOW/train27/weights/best.pt conf=0.25 source=
# https://www.youtube.com/watch?v=Fb-A88ho0sA
# https://www.youtube.com/watch?v=lZoZiqUFGlA
# https://www.youtube.com/watch?v=Cd43nNGvh00
# https://www.youtube.com/watch?v=w_qZ7vHabWI
# https://www.youtube.com/watch?v=VcywkVphqgY