from ultralytics import YOLO
import clearml

clearml.browser_login()

def freeze_layers(trainer):
    model = trainer.model
    num_freeze = 10  # Número de capas a congelar
    freeze = [f'model.{x}.' for x in range(num_freeze)]  # Capas a congelar

    for name, param in model.named_parameters():
        param.requires_grad = True  # Habilitar entrenamiento para todas las capas
        if any(x in name for x in freeze):
            param.requires_grad = False  # Congelar capas específicas

model = YOLO("models/yolov8n.pt")
model.add_callback("on_train_start", freeze_layers)

results = model.train(data="config.yaml", epochs=1, pretrained=True, workers=4)
