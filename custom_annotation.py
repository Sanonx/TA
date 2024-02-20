from ultralytics import YOLO
import cv2
import glob
import os
import shutil

# Directorio que contiene las imágenes
directorio = "youtube/videos/video_0/images/"

# Directorio para guardar las anotaciones
directorio_anotaciones = os.path.join(directorio, "labels")
os.makedirs(directorio_anotaciones, exist_ok=True)

# Directorio para guardar las imágenes
directorio_imagenes = os.path.join(directorio, "images")
os.makedirs(directorio_imagenes, exist_ok=True)

# Obtener una lista de rutas de imágenes en el directorio
rutas_imagenes = glob.glob(directorio + "/*.jpg") 

# Realizar la inferencia con el modelo preentrenado
model = YOLO("runsMLFLOW/train28/weights/best.pt")

# Iterar sobre todas las rutas de las imágenes
for ruta_imagen in rutas_imagenes:
    # Realizar la inferencia
    results = model.predict(source=ruta_imagen)

    # Obtener el nombre de la imagen sin la extensión
    nombre_imagen = os.path.basename(ruta_imagen)[:-4]

    # Si no se detectaron anotaciones, eliminar la imagen
    if len(results[0].boxes) == 0:
        os.remove(ruta_imagen)
    else:
        # Crear el nombre del archivo de anotaciones
        ruta_anotaciones = os.path.join(directorio_anotaciones, nombre_imagen + ".txt")

        # Guardar las anotaciones en formato YOLO
        with open(ruta_anotaciones, "w") as f:
            # Accediendo al primer (y único) objeto de Results en la lista
            res = results[0]
            
            for box, cls in zip(res.boxes.xywhn, res.boxes.cls):
                x, y, w, h = box.numpy()
                f.write(f"{int(cls.item())} {x} {y} {w} {h}\n")

        # Mover la imagen al directorio /images
        ruta_nueva_imagen = os.path.join(directorio_imagenes, os.path.basename(ruta_imagen))
        shutil.move(ruta_imagen, ruta_nueva_imagen)
