from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import random
import glob

# Directorio que contiene las imágenes
directorio = "NewDataset/Person2/images/test"

# Obtener una lista de rutas de imágenes en el directorio
rutas_imagenes = glob.glob(directorio + "/*.jpg")

# Elegir aleatoriamente 5 imágenes
rutas_imagenes_aleatorias = random.sample(rutas_imagenes, 5)

# Crear una figura con 5 subplots para mostrar las imágenes
fig, axs = plt.subplots(1, 5, figsize=(20, 20))

# Iterar sobre las rutas de las imágenes aleatorias
for i, ruta_imagen in enumerate(rutas_imagenes_aleatorias):
    # Realizar la inferencia con el modelo preentrenado
    model = YOLO("train15/weights/best.pt")
    results = model.predict(source=ruta_imagen)
    
    # Obtener la imagen con las cajas del resultado
    res_plotted = results[0].plot()
    
    # Convertir la imagen a formato RGB
    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    
    # Mostrar la imagen en el subplot correspondiente
    axs[i].imshow(res_rgb)
    axs[i].axis('off')

# Mostrar la figura con las imágenes
plt.show()

