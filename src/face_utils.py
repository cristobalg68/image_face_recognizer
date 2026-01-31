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


def extract_faces(image, detections):
    """
    Extrae las regiones de caras de la imagen
    
    Args:
        image: Imagen original (numpy array)
        detections: Lista de detecciones procesadas
    """
    h, w = image.shape[:2]
    
    for det in detections:
        x1, y1, x2, y2 = det['xyxy']
        
        # Asegurar que las coordenadas estén dentro de la imagen
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Extraer cara
        face = image[y1:y2, x1:x2]
        
        if face.size > 0:
            # Redimensionar a tamaño estándar para hash consistente
            face_resized = cv2.resize(face, (200, 200))
            det['face_image'] = face_resized


def hash_image(img, hash_size):
    """
    Calcula hash combinado (dhash + phash) de una imagen
    
    Args:
        img: Imagen (numpy array BGR)
        hash_size: Tamaño del hash
        
    Returns:
        String con hash combinado
    """
    # Convertir BGR a RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Calcular hashes
    dhash = str(imagehash.dhash(pil_img, hash_size))
    phash = str(imagehash.phash(pil_img, hash_size))
    
    return f'{dhash}{phash}'


def hash_faces(detections, hash_size):
    """
    Calcula hashes para todas las caras detectadas
    
    Args:
        detections: Lista de detecciones con face_image
        hash_size: Tamaño del hash
    """
    for det in detections:
        if 'face_image' in det:
            # Hash normal
            det['hash'] = hash_image(det['face_image'], hash_size)
            
            # Hash volteado (espejo)
            flipped = cv2.flip(det['face_image'], 1)
            det['hash_flipped'] = hash_image(flipped, hash_size)


def match_faces(detections, db, threshold=102.0):
    """
    Busca coincidencias de caras en la base de datos
    
    Args:
        detections: Lista de detecciones con hashes
        db: Instancia de FaceDatabase
        threshold: Umbral de similitud (default: 102.0)
    """
    for det in detections:
        if 'hash' in det:
            # Buscar coincidencia para hash normal
            match1, sim1, img_path1 = db.find_match(det['hash'], threshold)
            
            # Buscar coincidencia para hash volteado
            match2, sim2, img_path2 = db.find_match(det['hash_flipped'], threshold)
            
            # Elegir la mejor coincidencia
            if match1 and (not match2 or sim1 >= sim2):
                det['person_name'] = match1
                det['similarity'] = sim1
                det['db_image_path'] = img_path1
                det['is_unknown'] = False
            elif match2:
                det['person_name'] = match2
                det['similarity'] = sim2
                det['db_image_path'] = img_path2
                det['is_unknown'] = False
            else:
                # No se encontró coincidencia
                det['person_name'] = 'Desconocido'
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


def track_faces(detections, tracker, iou_threshold=0.5):
    """
    Rastrea caras entre frames para mantener IDs consistentes
    
    Args:
        detections: Nuevas detecciones del frame actual
        tracker: Diccionario con estado del tracker
        iou_threshold: Umbral de IoU para considerar match
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
        
        # Si encontramos match, actualizar
        if best_det_idx >= 0:
            matched_detections.add(best_det_idx)
            det = detections[best_det_idx]
            
            # Actualizar información del track
            new_faces[track_id] = {
                'bbox': det['bbox'],
                'xyxy': det['xyxy'],
                'face_image': det.get('face_image'),
                'person_name': det.get('person_name'),
                'similarity': det.get('similarity', 0),
                'db_image_path': det.get('db_image_path'),
                'is_unknown': det.get('is_unknown', False)
            }
    
    # Agregar nuevas detecciones que no hicieron match
    for i, det in enumerate(detections):
        if i not in matched_detections:
            track_id = str(tracker['last_id'])
            
            new_faces[track_id] = {
                'bbox': det['bbox'],
                'xyxy': det['xyxy'],
                'face_image': det.get('face_image'),
                'person_name': det.get('person_name'),
                'similarity': det.get('similarity', 0),
                'db_image_path': det.get('db_image_path'),
                'is_unknown': det.get('is_unknown', False)
            }
            
            tracker['last_id'] += 1
    
    # Actualizar tracker
    tracker['faces'] = new_faces


def draw_faces(image, tracker):
    """
    Dibuja las caras detectadas en la imagen con nombres
    
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
        
        if is_unknown:
            label = person_name
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
