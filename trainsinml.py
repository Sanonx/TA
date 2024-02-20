import os
from random import random, randint
from ultralytics import YOLO
import torch.nn as nn
import torch.nn.functional as F



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


    # Establecer el nombre de la ejecución

    # Crear una instancia de tu modelo
    model = YOLO("models/yolov8x.pt")

    # Crear instancia de la pérdida focal
    focal_loss = FocalLoss(alpha=0.25,gamma=1.5)


    model.loss = focal_loss

    # Añadir el callback a tu modelo


    # Entrenar el modelo
    results = model.train(
                        data="config.yaml", epochs=35, pretrained=True, workers=7, device=0, 
                        batch=8, patience=0, seed=42, cos_lr=True, imgsz=(640,480),
                        lr0=0.01, lrf=0.0001, momentum=0.937, weight_decay=0.001, warmup_epochs=0,
                        warmup_momentum=0.8, warmup_bias_lr=0, box=7.5, cls=0.5, dfl=1.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                        degrees=0.0, translate=0.1, scale=0.5, shear=0, perspective=0.0, flipud=0.0, fliplr=0.5,
                        mosaic=1.0, mixup=0.0, copy_paste=0.0, 
                        )

    
