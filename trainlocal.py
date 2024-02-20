import os
import mlflow
from random import random, randint
from ultralytics import YOLO
from ultralytics.yolo.utils.loss import FocalLoss
from ultralytics.yolo.utils.callbacks.mlflow import callbacks, on_fit_epoch_end, on_pretrain_routine_end, on_train_end
import subprocess
from ultralytics.yolo.data.augment import Albumentations

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
os.environ["MLFLOW_EXPERIMENT"] = "TeamAware"

mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "TeamAware"
mlflow.set_experiment(experiment_name)


if __name__ == "__main__":
    
    # Iniciar el registro de MLflow
    with mlflow.start_run(tags={"version1":"v1"}) as run:

        # Establecer el nombre de la ejecución

        # Crear una instancia de tu modelo
        model = YOLO("models/yolov8n.pt")

        # Crear instancia de la pérdida focal
        focal_loss = FocalLoss()

        # Asignar la pérdida focal al modelo
        model.loss = focal_loss

        
        # Añadir el callback a tu modelo
        # Agrega tus funciones de callback al modelo
        model.add_callback('on_pretrain_routine_end', on_pretrain_routine_end)
        model.add_callback('on_fit_epoch_end', on_fit_epoch_end)
        model.add_callback('on_train_end', on_train_end)

        # Entrenar el modelo
        results = model.train(data='config_muestra.yaml', epochs=2)

        # Guardar los resultados en MLflow
        # Supongamos que el modelo entrenado se guarda en 'model.pt'

        # Guardar los resultados en MLflow
        mlflow.log_artifact('config_muestra.yaml', "artifacts")
        mlflow.log_artifact(__file__, "artifacts")

        # Finalizar el registro de MLflow
        mlflow.end_run()
