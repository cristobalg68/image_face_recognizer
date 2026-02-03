import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO


class ImprovedDetector:

    def __init__(self, path_weights, min_confidence=0.6, min_face_size=40, max_aspect_ratio=2.5):
        """
        Inicializa el detector mejorado
        
        Args:
            path_weights: Ruta al archivo de pesos del modelo YOLO
            min_confidence: Confianza mínima para aceptar detección (default: 0.6)
            min_face_size: Tamaño mínimo en píxeles para la cara (default: 40)
            max_aspect_ratio: Ratio máximo ancho/alto permitido (default: 2.5)
        """
        if not os.path.exists(path_weights):
            print("\n" + "="*70)
            print(f"ERROR: No se encontró el modelo YOLO")
            print("="*70)
            print(f"\nRuta buscada: {path_weights}")
            sys.exit(1)
        
        try:
            self.model = YOLO(path_weights)
            print(f"Modelo YOLO cargado: {path_weights}")
        except Exception as e:
            print(f"\n Error al cargar el modelo YOLO: {e}")
            sys.exit(1)
        
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size
        self.max_aspect_ratio = max_aspect_ratio
        
        # Estadísticas de filtrado
        self.stats = {
            'total_detections': 0,
            'filtered_confidence': 0,
            'filtered_size': 0,
            'filtered_aspect': 0,
            'filtered_skin': 0,
            'valid_detections': 0
        }

    def detect_objects(self, img, conf, iou):
        """
        Detecta caras en una imagen con filtros de validación
        
        Args:
            img: Imagen a procesar
            conf: Umbral de confianza
            iou: Umbral de IoU para NMS
            
        Returns:
            Resultados de la detección (solo caras válidas)
        """
        # Detección básica con YOLO
        results = self.model(img, conf=conf, iou=iou, verbose=False)
        
        # Aplicar filtros adicionales
        filtered_results = self._filter_detections(results[0], img)
        
        return [filtered_results]
    
    def _filter_detections(self, results, img):
        """
        Filtra detecciones para reducir falsos positivos
        
        Args:
            results: Resultados crudos de YOLO
            img: Imagen original
            
        Returns:
            Resultados filtrados
        """
        if results.boxes is None or len(results.boxes) == 0:
            return results
        
        valid_indices = []
        
        for i, box in enumerate(results.boxes):
            self.stats['total_detections'] += 1
            
            # Obtener datos de la caja
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Filtro 1: Confianza mínima
            if confidence < self.min_confidence:
                self.stats['filtered_confidence'] += 1
                continue
            
            # Filtro 2: Tamaño mínimo
            width = x2 - x1
            height = y2 - y1
            
            if width < self.min_face_size or height < self.min_face_size:
                self.stats['filtered_size'] += 1
                continue
            
            # Filtro 3: Aspect ratio razonable (las caras no son muy alargadas)
            aspect_ratio = max(width / height, height / width)
            if aspect_ratio > self.max_aspect_ratio:
                self.stats['filtered_aspect'] += 1
                continue
            
            # Filtro 4: Detección de tonos de piel (básico)
            face_region = img[y1:y2, x1:x2]
            if not self._has_skin_tone(face_region):
                self.stats['filtered_skin'] += 1
                continue
            
            # Si pasó todos los filtros, es válida
            valid_indices.append(i)
            self.stats['valid_detections'] += 1
        
        # Crear nuevo resultado solo con detecciones válidas
        if len(valid_indices) > 0:
            # Filtrar boxes
            valid_boxes = results.boxes[valid_indices]
            results.boxes = valid_boxes
        else:
            # No hay detecciones válidas
            results.boxes = None
        
        return results
    
    def _has_skin_tone(self, face_region, min_skin_percentage=10.0):
        """
        Verifica si la región contiene tonos de piel
        
        Args:
            face_region: Región de la imagen a analizar
            min_skin_percentage: Porcentaje mínimo de píxeles con tono de piel
            
        Returns:
            True si tiene suficientes píxeles de tono de piel
        """
        if face_region.size == 0:
            return False
        
        try:
            # Convertir a YCrCb (mejor para detección de piel)
            ycrcb = cv2.cvtColor(face_region, cv2.COLOR_BGR2YCrCb)
            
            # Rangos de tono de piel en YCrCb (funciona para diversos tonos)
            lower_skin = np.array([0, 133, 77], dtype=np.uint8)
            upper_skin = np.array([255, 173, 127], dtype=np.uint8)
            
            # Crear máscara de piel
            skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
            
            # Calcular porcentaje de piel
            skin_pixels = np.count_nonzero(skin_mask)
            total_pixels = face_region.shape[0] * face_region.shape[1]
            skin_percentage = (skin_pixels / total_pixels) * 100
            
            return skin_percentage >= min_skin_percentage
            
        except Exception as e:
            # En caso de error, asumir que es válida
            return True
    
    def get_stats(self):
        """Retorna estadísticas de filtrado"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Resetea las estadísticas"""
        self.stats = {
            'total_detections': 0,
            'filtered_confidence': 0,
            'filtered_size': 0,
            'filtered_aspect': 0,
            'filtered_skin': 0,
            'valid_detections': 0
        }


# Mantener compatibilidad con código anterior
class Detector(ImprovedDetector):
    """Alias para compatibilidad con código existente"""
    pass
