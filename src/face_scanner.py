import cv2
import numpy as np
from detector import Detector
from face_database import FaceDatabase
import face_utils as utils


class FaceScanner:
    """Clase base para escáneres de reconocimiento facial"""

    def __init__(self, path_weights, size, confidence, iou, hash_size, db_path):
        """
        Args:
            path_weights: Ruta a los pesos del modelo YOLO
            size: Tamaño de redimensionamiento
            confidence: Umbral de confianza
            iou: Umbral de IOU
            hash_size: Tamaño del hash
            db_path: Ruta a la base de datos SQLite
        """
        self.detector = Detector(path_weights)
        self.size = size
        self.confidence = confidence
        self.iou = iou
        self.hash_size = hash_size
        
        # Inicializar base de datos
        self.db = FaceDatabase(db_path, images_cache_dir="data/face_images")
        self.db.connect()


class LiveFaceScanner(FaceScanner):
    """Scanner para reconocimiento facial en vivo desde webcam"""

    def __init__(self, path_weights, size, confidence, iou, hash_size, db_path):
        super().__init__(path_weights, size, confidence, iou, hash_size, db_path)
        self.tracker = {
            'last_id': 0,
            'faces': {}  # Diccionario de caras rastreadas
        }
        self.detected_persons = {}  # Personas únicas detectadas {person_id: data}

    def process_frame(self, frame):
        """
        Procesa un frame de la webcam para detección facial
        
        Args:
            frame: numpy array (BGR) del frame a procesar
            
        Returns:
            frame procesado con las detecciones dibujadas
        """
        if frame is None:
            return None

        # Hacer copia para dibujar
        frame_copy = frame.copy()
        
        # Redimensionar para procesamiento
        h_original, w_original = frame.shape[:2]
        img_resized = cv2.resize(frame, (self.size, self.size))
        
        # Detectar caras
        detections = self.detector.detect_objects(img_resized, self.confidence, self.iou)[0]
        detections = utils.process_detections(detections)
        
        # Extraer caras y calcular hashes
        utils.extract_faces(img_resized, detections)
        utils.hash_faces(detections, self.hash_size)
        
        # Buscar coincidencias en la base de datos
        utils.match_faces(detections, self.db)
        
        # Tracking de caras
        utils.track_faces(detections, self.tracker, self.iou)
        
        # Actualizar personas detectadas (sin duplicados)
        self._update_detected_persons()
        
        # Escalar coordenadas al tamaño original
        scale_x = w_original / self.size
        scale_y = h_original / self.size
        
        tracker_scaled = {'faces': {}}
        for track_id, track_data in self.tracker['faces'].items():
            if 'bbox' in track_data:
                x, y, w, h = track_data['bbox']
                track_data_scaled = track_data.copy()
                track_data_scaled['bbox'] = [
                    int(x * scale_x),
                    int(y * scale_y),
                    int(w * scale_x),
                    int(h * scale_y)
                ]
                tracker_scaled['faces'][track_id] = track_data_scaled
        
        # Dibujar en el frame original
        utils.draw_faces(frame_copy, tracker_scaled)
        
        return frame_copy

    def _update_detected_persons(self):
        """Actualiza el diccionario de personas únicas detectadas"""
        for track_id, track_data in self.tracker['faces'].items():
            if 'person_name' in track_data:
                person_name = track_data['person_name']
                
                # Solo agregar si es nueva o actualizar si tiene mejor similitud
                if person_name not in self.detected_persons:
                    self.detected_persons[person_name] = track_data.copy()
                else:
                    # Actualizar si la similitud es mejor
                    current_sim = self.detected_persons[person_name].get('similarity', 0)
                    new_sim = track_data.get('similarity', 0)
                    if new_sim > current_sim:
                        self.detected_persons[person_name] = track_data.copy()

    def reset_tracker(self):
        """Resetea el tracker y las detecciones"""
        self.tracker = {
            'last_id': 0,
            'faces': {}
        }
        self.detected_persons = {}

    def get_detected_persons(self):
        """
        Retorna lista de personas únicas detectadas
        
        Returns:
            Lista de diccionarios con información de cada persona
        """
        persons = []
        for person_name, data in self.detected_persons.items():
            persons.append({
                'name': person_name,
                'similarity': data.get('similarity', 0),
                'face_image': data.get('face_image'),
                'db_image_path': data.get('db_image_path'),
                'is_unknown': data.get('is_unknown', False)
            })
        return persons

    def register_new_person(self, person_name, face_image):
        """
        Registra una nueva persona en la base de datos
        
        Args:
            person_name: Nombre de la persona
            face_image: Imagen de la cara (numpy array)
            
        Returns:
            True si se registró exitosamente
        """
        return self.db.register_person(person_name, face_image, self.hash_size)


class ImageFaceScanner(FaceScanner):
    """Scanner para reconocimiento facial en imágenes estáticas"""

    def process_image(self, file_path):
        """
        Procesa una imagen para detección y reconocimiento facial
        
        Args:
            file_path: Ruta a la imagen
            
        Returns:
            (imagen_procesada, lista_detecciones)
        """
        # Leer imagen
        img_original = utils.read_image(file_path, self.size)
        img_original_copy = img_original.copy()
        
        # Detectar caras
        detections = self.detector.detect_objects(img_original, self.confidence, self.iou)[0]
        detections = utils.process_detections(detections)

        # Extraer caras y calcular hashes
        utils.extract_faces(img_original, detections)
        utils.hash_faces(detections, self.hash_size)
        
        # Buscar coincidencias en la base de datos
        utils.match_faces(detections, self.db)

        # Dibujar resultados
        utils.draw_faces_simple(img_original_copy, detections)

        return img_original_copy, detections


class VideoFaceScanner(FaceScanner):
    """Scanner para reconocimiento facial en videos"""

    def __init__(self, path_weights, size, confidence, iou, hash_size, db_path):
        super().__init__(path_weights, size, confidence, iou, hash_size, db_path)
        self.tracker = {
            'last_id': 0,
            'faces': {}
        }
        self.detected_persons = {}

    def process_frame(self, frame):
        """
        Procesa un frame individual del video
        
        Args:
            frame: numpy array (BGR) del frame a procesar
            
        Returns:
            frame procesado con las detecciones dibujadas
        """
        if frame is None:
            return None
        
        # Redimensionar frame
        img_original = cv2.resize(frame, (self.size, self.size))
        img_original_copy = img_original.copy()

        # Detectar caras
        detections = self.detector.detect_objects(img_original, self.confidence, self.iou)[0]
        detections = utils.process_detections(detections)
        
        # Extraer caras y calcular hashes
        utils.extract_faces(img_original, detections)
        utils.hash_faces(detections, self.hash_size)
        utils.match_faces(detections, self.db)
        
        # Tracking de caras
        utils.track_faces(detections, self.tracker, self.iou)
        
        # Actualizar personas detectadas
        self._update_detected_persons()
        
        # Dibujar sobre el frame
        utils.draw_faces(img_original_copy, self.tracker)

        return img_original_copy

    def _update_detected_persons(self):
        """Actualiza el diccionario de personas únicas detectadas"""
        for track_id, track_data in self.tracker['faces'].items():
            if 'person_name' in track_data:
                person_name = track_data['person_name']
                
                if person_name not in self.detected_persons:
                    self.detected_persons[person_name] = track_data.copy()
                else:
                    current_sim = self.detected_persons[person_name].get('similarity', 0)
                    new_sim = track_data.get('similarity', 0)
                    if new_sim > current_sim:
                        self.detected_persons[person_name] = track_data.copy()

    def reset_tracker(self):
        """Resetea el tracker"""
        self.tracker = {
            'last_id': 0,
            'faces': {}
        }
        self.detected_persons = {}

    def get_detected_persons(self):
        """Retorna lista de personas únicas detectadas"""
        persons = []
        for person_name, data in self.detected_persons.items():
            persons.append({
                'name': person_name,
                'similarity': data.get('similarity', 0),
                'face_image': data.get('face_image'),
                'db_image_path': data.get('db_image_path'),
                'is_unknown': data.get('is_unknown', False)
            })
        return persons
