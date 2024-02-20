from ultralytics import YOLO
from ray import tune
import matplotlib.pyplot as plt

# Define un modelo YOLO
model = YOLO("yolov8n.pt")


result_grid = model.tune(
    data="/home/surones/TeamAware/config.yaml", 
    epochs=15, pretrained=True, workers=8, device=0,batch=32, patience=0, seed=42, cos_lr=True, imgsz=(640,480),
    space = {
        "lr0": tune.uniform(0.1, 0.001),
        "lrf": tune.choice([0.1]),
        "momentum": tune.choice([0.937]),
        "weight_decay": tune.choice([0.0015]),
        "warmup_momentum": tune.choice([0.8]),
        "warmup_bias_lr": tune.choice([0]), # Valor medio: 0.0 (aquí no hay un valor medio definido)
        "box": tune.choice([7.5]),
        "cls": tune.choice([0.3]),
        "dfl": tune.choice([1.5]),
        "warmup_epochs": tune.choice([3]),
        "hsv_h": tune.choice([0]),
        "hsv_s": tune.choice([0]),
        "hsv_v": tune.choice([0]),
        "degrees": tune.choice([0.0]),
        "translate": tune.choice([0]),
        "scale": tune.choice([0]),
        "shear": tune.choice([0]),
        "perspective": tune.choice([0.0]),
        "flipud": tune.choice([0.0]),
        "fliplr": tune.choice([0.0]),
        "mosaic": tune.choice([0.0]),
        "mixup": tune.choice([0.0]),
        "copy_paste": tune.choice([0.0])
    },

)
