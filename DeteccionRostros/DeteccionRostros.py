import cv2
import numpy as np

# Inicializar la cámara
camara = cv2.VideoCapture(0)

# Cargar los clasificadores para detectar caras y ojos
detector_caras = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
detector_ojos = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Bucle principal
while True:
    # Capturar imagen de la cámara
    exito, imagen = camara.read()
    
    # Obtener dimensiones de la imagen
    alto = imagen.shape[0]
    ancho = imagen.shape[1]
    
    # Calcular el centro de la imagen
    centro_x = ancho / 2
    centro_y = alto / 2
    
    # Convertir imagen a escala de grises (necesario para detección)
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Detectar caras en la imagen
    caras_detectadas = detector_caras.detectMultiScale(imagen_gris, 1.3, 5)
    
    # Procesar cada cara detectada
    for (cara_x, cara_y, cara_ancho, cara_alto) in caras_detectadas:
        
        # Dibujar rectángulo verde alrededor de la cara
        cv2.rectangle(imagen, (cara_x, cara_y), (cara_x + cara_ancho, cara_y + cara_alto), (0, 255, 0), 3)
        
        # Escribir texto "Rostro" encima del rectángulo
        cv2.putText(imagen, "Rostro", (cara_x, cara_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mostrar posición de la cara
        cv2.putText(imagen, f"Posicion: [{cara_x}, {cara_y}]", (cara_x, cara_y - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Calcular centro de la cara
        centro_cara_x = cara_x + (cara_ancho / 2)
        centro_cara_y = cara_y + (cara_alto / 2)
        
        # Calcular distancia desde el centro de la imagen al centro de la cara
        distancia_x = centro_cara_x - centro_x
        distancia_y = centro_cara_y - centro_y
        distancia_total = np.sqrt(distancia_x**2 + distancia_y**2)
        
        # Mostrar la distancia en pantalla
        cv2.putText(imagen, f"Dist: {distancia_total:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Dibujar un punto rojo en el centro de la cara
        centro_cara_x_int = int(centro_cara_x)
        centro_cara_y_int = int(centro_cara_y)
        cv2.circle(imagen, (centro_cara_x_int, centro_cara_y_int), 5, (0, 0, 255), -1)
        
        # Calcular área de la cara
        area_cara = cara_ancho * cara_alto
        cv2.putText(imagen, f"Area: {area_cara}", (cara_x, cara_y - 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 255, 45), 1)
        
        # Extraer región de interés (ROI) - solo el área de la cara
        region_cara_gris = imagen_gris[cara_y:cara_y + cara_alto, cara_x:cara_x + cara_ancho]
        region_cara_color = imagen[cara_y:cara_y + cara_alto, cara_x:cara_x + cara_ancho]
        
        # Detectar ojos dentro de la región de la cara
        ojos_detectados = detector_ojos.detectMultiScale(region_cara_gris, 1.3, 5)
        
        # Procesar cada ojo detectado
        for (ojo_x, ojo_y, ojo_ancho, ojo_alto) in ojos_detectados:
            
            # Dibujar rectángulo azul alrededor del ojo en la imagen original
            cv2.rectangle(imagen, (cara_x + ojo_x, cara_y + ojo_y), 
                         (cara_x + ojo_x + ojo_ancho, cara_y + ojo_y + ojo_alto), (255, 0, 0), 1)
            
            # Calcular posición relativa del ojo dentro de la cara (0.0 a 1.0)
            posicion_relativa_x = ojo_x / cara_ancho
            posicion_relativa_y = ojo_y / cara_alto
            
            # Mostrar posición relativa del ojo
            cv2.putText(imagen, f"[{posicion_relativa_x:.2f}, {posicion_relativa_y:.2f}]", 
                       (cara_x + ojo_x, cara_y + ojo_y + ojo_alto + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 255, 0), 1)
            
            # Dibujar rectángulo adicional en la región de color
            cv2.rectangle(region_cara_color, (ojo_x, ojo_y), 
                         (ojo_x + ojo_ancho, ojo_y + ojo_alto), (430, 160, 0), 2)
    
    # Mostrar la imagen con las detecciones
    cv2.imshow("Deteccion De Rostros Y Ojos", imagen)
    
    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) == ord("q"):
        break

# Liberar recursos
camara.release()
cv2.destroyAllWindows()
