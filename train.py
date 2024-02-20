import os
import mlflow
from random import random, randint
from ultralytics import YOLO
import torch.nn as nn
from ultralytics.yolo.utils.callbacks.mlflow import callbacks, on_fit_epoch_end, on_pretrain_routine_end, on_train_end
import subprocess
import torch.nn.functional as F


os.environ["MLFLOW_TRACKING_URI"] = "http://34.250.123.180:9080"
os.environ["MLFLOW_EXPERIMENT"] = "TeamAware"

mlflow.set_tracking_uri("http://34.250.123.180:9080")
experiment_name = "TeamAware"
mlflow.set_experiment(experiment_name)

class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn()."""

    def __init__(self, alpha=0.25, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, label):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if self.alpha > 0:
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()



if __name__ == "__main__":

    # Iniciar el registro de MLflow
    with mlflow.start_run(tags={"version":"v1"}) as run:

        # Establecer el nombre de la ejecución
        mlflow.set_tag("mlflow.runName", "alldata")

        # Crear una instancia de tu modelo
        model = YOLO("models/yolov8x.pt")

        # Crear instancia de la pérdida focal
        focal_loss = FocalLoss(alpha=0.25,gamma=1.5)


        model.loss = focal_loss

        # Añadir el callback a tu modelo
        # Agrega tus funciones de callback al modelo
        model.add_callback('on_pretrain_routine_end', on_pretrain_routine_end)
        model.add_callback('on_fit_epoch_end', on_fit_epoch_end)
        model.add_callback('on_train_end', on_train_end)

        # Entrenar el modelo
        results = model.train(
                            data="config.yaml", epochs=35, pretrained=True, workers=7, device=0, 
                            batch=8, patience=0, seed=42, cos_lr=True, imgsz=(640,480),
                            lr0=0.01, lrf=0.0001, momentum=0.937, weight_decay=0.001, warmup_epochs=0,
                            warmup_momentum=0.8, warmup_bias_lr=0, box=7.5, cls=0.5, dfl=1.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                            degrees=0.0, translate=0.1, scale=0.5, shear=0, perspective=0.0, flipud=0.0, fliplr=0.5,
                            mosaic=1.0, mixup=0.0, copy_paste=0.0, 
                            )

        # Guardar los resultados en MLflow
        mlflow.log_artifact('config.yaml', artifact_path="")

        # Guardar el archivo .py
        mlflow.log_artifact(__file__, artifact_path="")

        # Finalizar el registro de MLflow
        mlflow.end_run()
