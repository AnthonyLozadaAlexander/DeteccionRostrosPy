import cv2
import numpy as np

# MEJORA: Verificar disponibilidad de cámara
cap = cv2.VideoCapture(0) # 0 para la cámara predeterminada, 1 para una cámara externa
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()

cara_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # Cargamos el clasificador preentrenado para detección de rostros

ojos_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml") # Cargamos el clasificador preentrenado para detección de ojos

# MEJORA: Verificar que los clasificadores se cargaron correctamente
if cara_cascade.empty() or ojos_cascade.empty():
    print("Error: No se pudieron cargar los clasificadores")
    exit()

while True:
    # Lee un frame de la cámara
    ret, frame = cap.read()
    
    # MEJORA: Verificar que se leyó el frame correctamente
    if not ret:
        print("Error: No se pudo leer el frame")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convertimos la imagen a escala de grises
    
    # OPTIMIZACIÓN: Mejores parámetros para detección más precisa
    caras = cara_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30)) # Detectamos rostros en la imagen
    
    # CONCEPTOS DE ÁLGEBRA LINEAL APLICADOS:
    # x, y = coordenadas del punto superior izquierdo (origen del sistema de coordenadas)
    # w = ancho del rectángulo (extensión en el eje x)
    # h = alto del rectángulo (extensión en el eje y)
    # Las matrices se indexan como [filas, columnas] = [y:y+altura, x:x+ancho]
    
    for(x, y, w, h) in caras:

        # Rectangulo para el rostro
        cv2.putText(frame, "Rostro", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x,y), (x+w, y+h),(255,0,0), 2) 
        
        # CORRECCIÓN CRÍTICA: ROI (Region of Interest) correctamente definida
        # ERROR ORIGINAL: roi_gray = gray[y:y+w, x:x+w] (usaba ancho para altura)
        # CORRECCIÓN: En álgebra lineal, las matrices son [filas:filas+altura, columnas:columnas+ancho]
        roi_gray = gray[y:y+h, x:x+w]    # CORREGIDO: altura (h) para filas, ancho (w) para columnas
        roi_color = frame[y:y+h, x:x+w]  # CORREGIDO: dimensiones consistentes
        
        # OPTIMIZACIÓN: Mejores parámetros para detección de ojos
        ojos = ojos_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10)) # Detectamos ojos dentro del rostro detectado    
    
        for(ox, oy, ow, oh) in ojos:
            # MEJORA: Posición del texto más clara
            cv2.putText(frame, "Ojos", (x + ox, y + oy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            # CORRECCIÓN: Color verde para ojos (consistente con el texto)
            cv2.rectangle(roi_color, (ox,oy), (ox+ow, oy+oh),(0,255,0), 2) 
            
    # MEJORA: Información educativa en pantalla
    cv2.putText(frame, f"Rostros detectados: {len(caras)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Presiona 'q' para salir", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Deteccion De Rostros Y Ojos - Version Corregida", frame)
    
    # MEJORA: Uso correcto de máscara de bits para detectar tecla
    if(cv2.waitKey(1) & 0xFF == ord("q")):
        break

# MEJORA: Mensaje informativo al cerrar
print("Cerrando aplicación...")    
cap.release()
cv2.destroyAllWindows()
