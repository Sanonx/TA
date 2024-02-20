from ultralytics import YOLO
import clearml
from ultralytics.yolo.utils.loss import FocalLoss


clearml.browser_login()

model = YOLO("models/yolov8n.pt")

# Crear instancia de la pérdida focal
focal_loss = FocalLoss()

# Asignar la pérdida focal al modelo
model.loss = focal_loss


results = model.train(data="config.yaml", epochs=40, pretrained=True, workers=8, device=0, batch=32, patience=5,
		      seed=42, lr0=1E-2, lrf=1E-4)  # train the model





