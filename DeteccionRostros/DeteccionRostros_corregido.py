# DETECCIÓN DE ROSTROS Y OJOS - VERSIÓN CORREGIDA Y EDUCATIVA
# Profesor: Correcciones importantes para estudiante de programación junior
# Conceptos de álgebra lineal aplicados: matrices, sistemas de coordenadas, regiones de interés

import cv2
import numpy as np

# MEJORA 1: Verificar disponibilidad de cámara con manejo de errores
print("Iniciando captura de video...")
cap = cv2.VideoCapture(0)  # 0 para la cámara predeterminada, 1 para una cámara externa

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()

# MEJORA 2: Verificar que los clasificadores se carguen correctamente
cara_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
ojos_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Verificación de carga de clasificadores
if cara_cascade.empty():
    print("Error: No se pudo cargar el clasificador de rostros")
    exit()
if ojos_cascade.empty():
    print("Error: No se pudo cargar el clasificador de ojos")
    exit()

print("Clasificadores cargados correctamente. Presiona 'q' para salir.")

while True:
    # Lee un frame de la cámara
    ret, frame = cap.read()
    
    # MEJORA 3: Verificar que se leyó el frame correctamente
    if not ret:
        print("Error: No se pudo leer el frame de la cámara")
        break
    
    # Convertimos la imagen a escala de grises
    # CONCEPTO DE ÁLGEBRA LINEAL: Transformación de matriz 3D (BGR) a matriz 2D (escala de grises)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectamos rostros en la imagen
    # PARÁMETROS OPTIMIZADOS: scaleFactor=1.1 (más preciso), minNeighbors=6 (menos falsos positivos)
    caras = cara_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))
    
    # CONCEPTOS DE COORDENADAS Y DIMENSIONES (ÁLGEBRA LINEAL):
    # x, y = coordenadas del punto superior izquierdo del rectángulo (origen del sistema de coordenadas)
    # w = ancho del rectángulo (extensión en el eje x)
    # h = alto del rectángulo (extensión en el eje y)
    # Estos forman una matriz de transformación para definir la región de interés (ROI)
    
    for (x, y, w, h) in caras:
        # Dibujar rectángulo para el rostro (color azul en BGR: (255,0,0))
        cv2.putText(frame, "Rostro", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # CORRECCIÓN CRÍTICA: ROI (Region of Interest) correctamente definida
        # ERROR ORIGINAL: roi_gray = gray[y:y+w, x:x+w] (usaba ancho para altura)
        # CORRECCIÓN: En matrices, las dimensiones son [filas:filas+altura, columnas:columnas+ancho]
        # CONCEPTO DE ÁLGEBRA LINEAL: Submatriz extraída de la matriz principal
        roi_gray = gray[y:y+h, x:x+w]    # CORREGIDO: altura (h) para filas, ancho (w) para columnas
        roi_color = frame[y:y+h, x:x+w]  # CORREGIDO: dimensiones consistentes
        
        # Detectamos ojos dentro del rostro detectado (solo en la ROI)
        # OPTIMIZACIÓN: Parámetros ajustados para mejor detección en la región del rostro
        ojos = ojos_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))
        
        # Dibujar rectángulos para los ojos
        for (ox, oy, ow, oh) in ojos:
            # CORRECCIÓN: Posición del texto ajustada y color verde para ojos (0,255,0)
            cv2.putText(frame, "Ojos", (x + ox, y + oy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            # CORRECCIÓN: Color verde para ojos (no azul como en el original)
            cv2.rectangle(roi_color, (ox, oy), (ox + ow, oy + oh), (0, 255, 0), 2)
    
    # Mostrar información educativa en pantalla
    cv2.putText(frame, f"Rostros detectados: {len(caras)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Presiona 'q' para salir", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Mostrar el frame con las detecciones
    cv2.imshow("Deteccion De Rostros Y Ojos - Version Corregida", frame)
    
    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# MEJORA 4: Liberación segura de recursos
print("Cerrando aplicación...")
cap.release()
cv2.destroyAllWindows()