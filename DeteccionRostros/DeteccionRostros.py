import cv2
import numpy as np

cap = cv2.VideoCapture(0) # 0 para la cámara predeterminada, 1 para una cámara externa

cara_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # Cargamos el clasificador preentrenado para detección de rostros

ojos_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml") # Cargamos el clasificador preentrenado para detección de ojos

while True:
    # Lee un frame de la cámara
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convertimos la imagen a escala de grises
    
    caras = cara_cascade.detectMultiScale(gray, 1.3, 5) # Detectamos rostros en la imagen
    
    # x, y, w, h son las coordenadas y dimensiones del rectángulo que encierra el rostro detectado
    # x = coordenada x del rectángulo
    # y = coordenada y del rectángulo
    # w = ancho del rectángulo
    # h = alto del rectángulo
    
    for(x, y, w, h) in caras:

        # Rectangulo para el rostro
        cv2.putText(frame, "Rostro", (x, y-20), 2,0.5,(255,0,0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x,y), (x+w, y+h),(255,0,0), 5) 
        
        roi_gray = gray[y:y+w, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        ojos = ojos_cascade.detectMultiScale(roi_gray,1.3,5) # Detectamos ojos dentro del rostro detectado    
    
        for(ox, oy, ow, oh) in ojos:
            cv2.putText(frame, "Ojos", (x, y+60), 2,0.5,(0,255,0), 2, cv2.LINE_AA)
            cv2.rectangle(roi_color, (ox,oy), (ox+ow, oy+oh),(255,0,0), 5) 
            
    cv2.imshow("Deteccion De Rostros Y Ojos", frame)
    
    if(cv2.waitKey(1) == ord("q")):
        break
    
cap.release()
cv2.destroyAllWindows()
