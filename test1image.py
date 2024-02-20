from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
# Inferencia con un modelo preentrenado de una imagen seleccionada

model=YOLO("runsMLFLOW/train57/weights/best.pt")
results = model.predict(source='NewDataset/coco_reducido/images/valid/101906.jpg')

print(results[0].boxes)
res_plotted = results[0].plot()
# Convertir la imagen a formato RGB
res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

# Mostrar la imagen utilizando pyplot
plt.imshow(res_rgb)
plt.axis('off')
plt.show()