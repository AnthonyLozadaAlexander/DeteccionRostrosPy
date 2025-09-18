#!/usr/bin/env python3
"""
DEMOSTRACIÓN EDUCATIVA: Diferencia entre ROI incorrecta y correcta
Este programa muestra cómo el error en las dimensiones de ROI afecta la detección de ojos
"""

import cv2
import numpy as np

def crear_imagen_ejemplo():
    """Crea una imagen de ejemplo para demostrar el concepto de ROI"""
    # Crear una imagen de 400x400 píxeles
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Simular un "rostro" (rectángulo azul)
    x, y, w, h = 100, 150, 200, 180  # Coordenadas del rostro
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 100, 100), -1)
    
    # Simular "ojos" (círculos blancos) dentro del rostro
    cv2.circle(img, (x+50, y+50), 20, (255, 255, 255), -1)   # Ojo izquierdo
    cv2.circle(img, (x+150, y+50), 20, (255, 255, 255), -1)  # Ojo derecho
    
    # Agregar texto explicativo
    cv2.putText(img, "Rostro simulado", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, f"x={x}, y={y}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, f"w={w}, h={h}", (x, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img, (x, y, w, h)

def demostrar_roi_incorrecto(img, face_coords):
    """Demuestra el ROI incorrecto (como en el código original)"""
    x, y, w, h = face_coords
    
    # ERROR ORIGINAL: usar ancho para altura
    try:
        roi_incorrect = img[y:y+w, x:x+w]  # ❌ INCORRECTO
        cv2.imshow("ROI Incorrecto (Original)", roi_incorrect)
        print(f"❌ ROI Incorrecto: img[{y}:{y+w}, {x}:{x+w}] = img[{y}:{y+w}, {x}:{x+w}]")
        print(f"   Dimensiones resultantes: {roi_incorrect.shape}")
    except Exception as e:
        print(f"❌ ERROR en ROI incorrecto: {e}")
        # Mostrar imagen vacía si hay error
        error_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(error_img, "ERROR", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("ROI Incorrecto (Original)", error_img)

def demostrar_roi_correcto(img, face_coords):
    """Demuestra el ROI correcto (versión corregida)"""
    x, y, w, h = face_coords
    
    # CORRECCIÓN: usar altura para filas, ancho para columnas
    roi_correct = img[y:y+h, x:x+w]  # ✅ CORRECTO
    cv2.imshow("ROI Correcto (Corregido)", roi_correct)
    print(f"✅ ROI Correcto: img[{y}:{y+h}, {x}:{x+w}] = img[{y}:{y+h}, {x}:{x+w}]")
    print(f"   Dimensiones resultantes: {roi_correct.shape}")

def main():
    print("=" * 60)
    print("DEMOSTRACIÓN EDUCATIVA: ROI Correcto vs Incorrecto")
    print("=" * 60)
    
    # Crear imagen de ejemplo
    img, face_coords = crear_imagen_ejemplo()
    x, y, w, h = face_coords
    
    print(f"\nCoordenadas del rostro simulado:")
    print(f"x (columna inicial) = {x}")
    print(f"y (fila inicial) = {y}")
    print(f"w (ancho) = {w}")
    print(f"h (alto) = {h}")
    print(f"Tamaño de imagen completa: {img.shape}")
    
    # Mostrar imagen original
    img_copy = img.copy()
    cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Imagen Original con Rostro Detectado", img_copy)
    
    print("\n" + "="*60)
    print("COMPARACIÓN DE MÉTODOS:")
    print("="*60)
    
    # Demostrar ROI incorrecto
    print("\n1. MÉTODO ORIGINAL (INCORRECTO):")
    demostrar_roi_incorrecto(img, face_coords)
    
    # Demostrar ROI correcto
    print("\n2. MÉTODO CORREGIDO (CORRECTO):")
    demostrar_roi_correcto(img, face_coords)
    
    print("\n" + "="*60)
    print("CONCEPTOS DE ÁLGEBRA LINEAL:")
    print("="*60)
    print("• Las matrices se indexan como [filas, columnas]")
    print("• En imágenes: filas = coordenada Y, columnas = coordenada X")
    print("• ROI = Submatriz extraída de la matriz principal")
    print("• Fórmula correcta: img[y:y+altura, x:x+ancho]")
    print("\nPresiona cualquier tecla para cerrar...")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()