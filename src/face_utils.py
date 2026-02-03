import cv2
import numpy as np
import imagehash
from PIL import Image


def read_image(path_image, size):
    """Lee y redimensiona una imagen"""
    img = cv2.imread(path_image)
    img = cv2.resize(img, (size, size))
    return img


def process_detections(detections):
    """
    Procesa las detecciones de YOLO
    
    Args:
        detections: Resultados de YOLO
        
    Returns:
        Lista de diccionarios con información de detección
    """
    if detections.boxes is None or len(detections.boxes) == 0:
        return []
    
    processed = []
    for i, bbox in enumerate(detections.boxes.xywh):
        x, y, w, h = bbox.cpu().numpy()
        
        # Convertir de xywh a xyxy
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        
        processed.append({
            'bbox': [int(x), int(y), int(w), int(h)],  # xywh
            'xyxy': [x1, y1, x2, y2],  # xyxy para cropping
            'confidence': float(detections.boxes.conf[i].cpu().numpy())
        })
    
    return processed


def extract_faces(image, detections, padding=0.2):
    """
    Extrae las regiones de caras de la imagen con padding
    
    Args:
        image: Imagen original (numpy array)
        detections: Lista de detecciones procesadas
        padding: Porcentaje de padding alrededor de la cara (default: 0.2 = 20%)
    """
    h, w = image.shape[:2]
    
    for det in detections:
        x1, y1, x2, y2 = det['xyxy']
        
        # Agregar padding
        width = x2 - x1
        height = y2 - y1
        
        pad_w = int(width * padding)
        pad_h = int(height * padding)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        
        # Extraer cara
        face = image[y1:y2, x1:x2]
        
        if face.size > 0:
            # Redimensionar a tamaño estándar para hash consistente
            face_resized = cv2.resize(face, (200, 200))
            
            # Normalizar la imagen para mejor reconocimiento
            face_normalized = normalize_face(face_resized)
            
            det['face_image'] = face_normalized
            det['face_image_original'] = face_resized  # Guardar sin normalizar para visualización


def normalize_face(face_img):
    """
    Normaliza una imagen de cara para mejorar el reconocimiento
    Aplica corrección de iluminación y mejora de contraste
    
    Args:
        face_img: Imagen de la cara (numpy array BGR)
        
    Returns:
        Imagen normalizada
    """
    # Convertir a LAB para trabajar con luminosidad
    lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Ecualización adaptativa de histograma en canal L
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_equalized = clahe.apply(l)
    
    # Recombinar canales
    lab_equalized = cv2.merge([l_equalized, a, b])
    
    # Convertir de vuelta a BGR
    normalized = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)
    
    return normalized


def hash_image_multi(img, hash_size):
    """
    Calcula múltiples hashes para mejor robustez
    Incluye dhash, phash y ahash
    
    Args:
        img: Imagen (numpy array BGR)
        hash_size: Tamaño del hash
        
    Returns:
        String con hashes combinados
    """
    # Convertir BGR a RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Calcular múltiples tipos de hash
    dhash = str(imagehash.dhash(pil_img, hash_size))
    phash = str(imagehash.phash(pil_img, hash_size))
    ahash = str(imagehash.average_hash(pil_img, hash_size))
    
    # Combinar hashes
    return f'{dhash}{phash}{ahash}'


def hash_faces(detections, hash_size):
    """
    Calcula hashes para todas las caras detectadas con múltiples variaciones
    
    Args:
        detections: Lista de detecciones con face_image
        hash_size: Tamaño del hash
    """
    for det in detections:
        if 'face_image' in det:
            # Hash de imagen normalizada
            det['hash'] = hash_image_multi(det['face_image'], hash_size)
            
            # Hash volteado (espejo)
            flipped = cv2.flip(det['face_image'], 1)
            det['hash_flipped'] = hash_image_multi(flipped, hash_size)
            
            # Hash con rotaciones leves para mayor robustez
            # Rotación +5 grados
            rotated_5 = rotate_image(det['face_image'], 5)
            det['hash_rot5'] = hash_image_multi(rotated_5, hash_size)
            
            # Rotación -5 grados
            rotated_neg5 = rotate_image(det['face_image'], -5)
            det['hash_rot_neg5'] = hash_image_multi(rotated_neg5, hash_size)


def rotate_image(image, angle):
    """
    Rota una imagen un ángulo específico
    
    Args:
        image: Imagen a rotar
        angle: Ángulo en grados
        
    Returns:
        Imagen rotada
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Matriz de rotación
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotar
    rotated = cv2.warpAffine(image, M, (w, h), 
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    
    return rotated


def match_faces(detections, db, threshold=150.0):
    """
    Busca coincidencias de caras en la base de datos con búsqueda robusta
    Prueba múltiples variaciones de hash para mejor reconocimiento
    
    Args:
        detections: Lista de detecciones con hashes
        db: Instancia de FaceDatabase
        threshold: Umbral de similitud (default: 150.0 para hashes múltiples)
    """
    for det in detections:
        if 'hash' in det:
            best_match = None
            best_similarity = 0
            best_image_path = None
            
            # Probar todos los hashes disponibles
            hash_variants = [
                ('hash', det.get('hash', '')),
                ('hash_flipped', det.get('hash_flipped', '')),
                ('hash_rot5', det.get('hash_rot5', '')),
                ('hash_rot_neg5', det.get('hash_rot_neg5', ''))
            ]
            
            for variant_name, hash_value in hash_variants:
                if not hash_value:
                    continue
                
                match, sim, img_path = db.find_match(hash_value, threshold)
                
                if match and sim > best_similarity:
                    best_match = match
                    best_similarity = sim
                    best_image_path = img_path
            
            # Asignar mejor coincidencia
            if best_match:
                det['person_name'] = best_match
                det['similarity'] = best_similarity
                det['db_image_path'] = best_image_path
                det['is_unknown'] = False
            else:
                # No se encontró coincidencia - es desconocido
                det['person_name'] = None
                det['similarity'] = 0.0
                det['db_image_path'] = None
                det['is_unknown'] = True


def calcular_iou(bbox1, bbox2):
    """
    Calcula el IoU (Intersection over Union) entre dos bounding boxes
    
    Args:
        bbox1, bbox2: [x, y, w, h] en formato xywh
        
    Returns:
        Valor de IoU (0-1)
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Convertir a xyxy
    x1_min = x1 - w1/2
    y1_min = y1 - h1/2
    x1_max = x1 + w1/2
    y1_max = y1 + h1/2
    
    x2_min = x2 - w2/2
    y2_min = y2 - h2/2
    x2_max = x2 + w2/2
    y2_max = y2 + h2/2
    
    # Calcular intersección
    xA = max(x1_min, x2_min)
    yA = max(y1_min, y2_min)
    xB = min(x1_max, x2_max)
    yB = min(y1_max, y2_max)
    
    inter_ancho = max(0, xB - xA)
    inter_alto = max(0, yB - yA)
    inter_area = inter_ancho * inter_alto
    
    # Calcular áreas
    area1 = w1 * h1
    area2 = w2 * h2
    
    # Calcular unión
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    iou = inter_area / union_area
    return iou


def hamming_distance(hash1, hash2):
    """
    Calcula la distancia de Hamming entre dos hashes
    
    Args:
        hash1, hash2: Strings de hashes hexadecimales
        
    Returns:
        Distancia de Hamming (número de bits diferentes)
    """
    if not hash1 or not hash2 or len(hash1) != len(hash2):
        return float('inf')
    
    distance = 0
    for c1, c2 in zip(hash1, hash2):
        if c1 != c2:
            try:
                val1 = int(c1, 16)
                val2 = int(c2, 16)
                xor = val1 ^ val2
                distance += bin(xor).count('1')
            except ValueError:
                distance += 4
    
    return distance


def track_faces(detections, tracker, iou_threshold=0.5, unknown_similarity_threshold=80.0):
    """
    Rastrea caras entre frames para mantener IDs consistentes
    Asigna IDs únicos a cada cara desconocida diferente
    
    Args:
        detections: Nuevas detecciones del frame actual
        tracker: Diccionario con estado del tracker
        iou_threshold: Umbral de IoU para considerar match espacial
        unknown_similarity_threshold: Umbral de similitud de hash para caras desconocidas
    """
    new_faces = {}
    matched_detections = set()
    
    # Intentar hacer match con caras existentes
    for track_id, tracked_face in tracker['faces'].items():
        best_iou = 0
        best_det_idx = -1
        
        for i, det in enumerate(detections):
            if i in matched_detections:
                continue
            
            iou = calcular_iou(tracked_face['bbox'], det['bbox'])
            
            if iou > iou_threshold and iou > best_iou:
                best_iou = iou
                best_det_idx = i
        
        # Si encontramos match espacial, actualizar
        if best_det_idx >= 0:
            matched_detections.add(best_det_idx)
            det = detections[best_det_idx]
            
            # Actualizar información del track
            new_faces[track_id] = {
                'bbox': det['bbox'],
                'xyxy': det['xyxy'],
                'face_image': det.get('face_image_original', det.get('face_image')),
                'hash': det.get('hash'),
                'hash_flipped': det.get('hash_flipped'),
                'person_name': det.get('person_name') or tracked_face.get('person_name'),
                'similarity': det.get('similarity', 0),
                'db_image_path': det.get('db_image_path'),
                'is_unknown': det.get('is_unknown', False),
                'confidence': det.get('confidence', 0.0)
            }
    
    # Agregar nuevas detecciones que no hicieron match
    for i, det in enumerate(detections):
        if i not in matched_detections:
            # Determinar si es una cara desconocida nueva o similar a una existente
            if det.get('is_unknown', False):
                # Buscar si hay una cara desconocida similar
                similar_track_id = None
                min_distance = float('inf')
                
                det_hash = det.get('hash', '')
                
                for track_id, tracked_face in new_faces.items():
                    if tracked_face.get('is_unknown', False):
                        tracked_hash = tracked_face.get('hash', '')
                        
                        if det_hash and tracked_hash:
                            distance = hamming_distance(det_hash, tracked_hash)
                            
                            if distance < min_distance and distance < unknown_similarity_threshold:
                                min_distance = distance
                                similar_track_id = track_id
                
                # Si encontramos una cara desconocida similar, actualizar esa
                if similar_track_id is not None:
                    new_faces[similar_track_id] = {
                        'bbox': det['bbox'],
                        'xyxy': det['xyxy'],
                        'face_image': det.get('face_image_original', det.get('face_image')),
                        'hash': det.get('hash'),
                        'hash_flipped': det.get('hash_flipped'),
                        'person_name': new_faces[similar_track_id]['person_name'],
                        'similarity': 0.0,
                        'db_image_path': None,
                        'is_unknown': True,
                        'confidence': det.get('confidence', 0.0)
                    }
                    continue
                
                # Es una nueva cara desconocida
                track_id = str(tracker['last_id'])
                person_name = f"Desconocido #{tracker['last_id'] + 1}"
                
                new_faces[track_id] = {
                    'bbox': det['bbox'],
                    'xyxy': det['xyxy'],
                    'face_image': det.get('face_image_original', det.get('face_image')),
                    'hash': det.get('hash'),
                    'hash_flipped': det.get('hash_flipped'),
                    'person_name': person_name,
                    'similarity': 0.0,
                    'db_image_path': None,
                    'is_unknown': True,
                    'confidence': det.get('confidence', 0.0)
                }
                
                tracker['last_id'] += 1
            else:
                # Persona conocida
                track_id = str(tracker['last_id'])
                
                new_faces[track_id] = {
                    'bbox': det['bbox'],
                    'xyxy': det['xyxy'],
                    'face_image': det.get('face_image_original', det.get('face_image')),
                    'hash': det.get('hash'),
                    'hash_flipped': det.get('hash_flipped'),
                    'person_name': det.get('person_name'),
                    'similarity': det.get('similarity', 0),
                    'db_image_path': det.get('db_image_path'),
                    'is_unknown': False,
                    'confidence': det.get('confidence', 0.0)
                }
                
                tracker['last_id'] += 1
    
    # Actualizar tracker
    tracker['faces'] = new_faces


def draw_faces(image, tracker):
    """
    Dibuja las caras detectadas en la imagen con nombres y confianza
    
    Args:
        image: Imagen donde dibujar
        tracker: Diccionario con caras rastreadas
    """
    for track_id, face_data in tracker['faces'].items():
        x, y, w, h = face_data['bbox']
        
        # Calcular coordenadas del rectángulo
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        
        # Determinar color basado en si es conocido o desconocido
        is_unknown = face_data.get('is_unknown', True)
        if is_unknown:
            color = (0, 165, 255)  # Naranja para desconocidos
        else:
            color = (0, 255, 0)  # Verde para conocidos
        
        # Dibujar rectángulo
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Preparar texto
        person_name = face_data.get('person_name', 'Desconocido')
        similarity = face_data.get('similarity', 0)
        confidence = face_data.get('confidence', 0)
        
        if is_unknown:
            label = f"{person_name} ({confidence*100:.0f}%)"
        else:
            label = f"{person_name} ({similarity:.0f}%)"
        
        # Dibujar fondo para el texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        
        # Rectángulo de fondo
        cv2.rectangle(
            image,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 10, y1),
            color,
            -1
        )
        
        # Texto
        cv2.putText(
            image,
            label,
            (x1 + 5, y1 - 5),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )


def draw_faces_simple(image, detections):
    """
    Dibuja caras detectadas sin tracking (para imágenes estáticas)
    
    Args:
        image: Imagen donde dibujar
        detections: Lista de detecciones
    """
    for det in detections:
        x, y, w, h = det['bbox']
        
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        
        is_unknown = det.get('is_unknown', True)
        color = (0, 165, 255) if is_unknown else (0, 255, 0)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        person_name = det.get('person_name', 'Desconocido')
        similarity = det.get('similarity', 0)
        
        if is_unknown:
            label = person_name
        else:
            label = f"{person_name} ({similarity:.0f}%)"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        
        cv2.rectangle(
            image,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 10, y1),
            color,
            -1
        )
        
        cv2.putText(
            image,
            label,
            (x1 + 5, y1 - 5),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )
