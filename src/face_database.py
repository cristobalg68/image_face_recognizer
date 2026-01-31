import sqlite3
import os
import shutil
from pathlib import Path
import imagehash
from PIL import Image
import cv2
import numpy as np


class FaceDatabase:
    """
    Gestor de base de datos para reconocimiento facial
    Almacena personas registradas con sus hashes faciales
    """
    
    def __init__(self, db_path, images_cache_dir="data/face_images"):
        self.db_path = db_path
        self.images_cache_dir = images_cache_dir
        self.conn = None
        
        # Crear directorios si no existen
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(images_cache_dir, exist_ok=True)
    
    def connect(self):
        """Conecta a la base de datos y crea las tablas si no existen"""
        self.conn = sqlite3.connect(self.db_path)
        self._create_tables()
        return self.conn
    
    def _create_tables(self):
        """Crea las tablas necesarias"""
        cursor = self.conn.cursor()
        
        # Tabla de personas registradas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                hash_normal TEXT NOT NULL,
                hash_flipped TEXT NOT NULL,
                image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Índices para búsqueda rápida
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_hash_normal 
            ON persons(hash_normal)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_hash_flipped 
            ON persons(hash_flipped)
        ''')
        
        self.conn.commit()
    
    def register_person(self, name, face_image, hash_size=16):
        """
        Registra una nueva persona en la base de datos
        
        Args:
            name: Nombre de la persona
            face_image: Imagen de la cara (numpy array o PIL Image)
            hash_size: Tamaño del hash para perceptual hashing
            
        Returns:
            True si se registró exitosamente, False si ya existe
        """
        cursor = self.conn.cursor()
        
        # Verificar si ya existe
        cursor.execute('SELECT id FROM persons WHERE name = ?', (name,))
        if cursor.fetchone():
            return False
        
        # Convertir a PIL Image si es necesario
        if isinstance(face_image, np.ndarray):
            # Convertir de BGR a RGB si viene de OpenCV
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_image)
        else:
            pil_image = face_image
        
        # Calcular hashes
        hash_normal = self._compute_hash(pil_image, hash_size)
        
        # Crear versión volteada y calcular hash
        flipped = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
        hash_flipped = self._compute_hash(flipped, hash_size)
        
        # Guardar imagen en caché
        image_filename = f"{name.replace(' ', '_').lower()}.jpg"
        image_path = os.path.join(self.images_cache_dir, image_filename)
        pil_image.save(image_path, 'JPEG', quality=95)
        
        # Insertar en la base de datos
        cursor.execute('''
            INSERT INTO persons (name, hash_normal, hash_flipped, image_path)
            VALUES (?, ?, ?, ?)
        ''', (name, hash_normal, hash_flipped, image_path))
        
        self.conn.commit()
        return True
    
    def _compute_hash(self, pil_image, hash_size):
        """Calcula hash combinado (dhash + phash)"""
        pil_rgb = pil_image.convert('RGB')
        dhash = str(imagehash.dhash(pil_rgb, hash_size))
        phash = str(imagehash.phash(pil_rgb, hash_size))
        return f"{dhash}{phash}"
    
    def find_match(self, face_hash, threshold=102.0):
        """
        Busca coincidencia en la base de datos
        
        Args:
            face_hash: Hash de la cara a buscar
            threshold: Umbral de similitud (distancia Hamming máxima)
            
        Returns:
            (nombre, similitud, ruta_imagen) o (None, None, None)
        """
        cursor = self.conn.cursor()
        
        # Obtener todas las personas
        cursor.execute('SELECT name, hash_normal, hash_flipped, image_path FROM persons')
        persons = cursor.fetchall()
        
        best_match = None
        best_similarity = float('inf')
        best_image_path = None
        
        for name, hash_normal, hash_flipped, image_path in persons:
            # Calcular similitud con hash normal
            sim_normal = self._hamming_distance(face_hash, hash_normal)
            
            # Calcular similitud con hash volteado
            sim_flipped = self._hamming_distance(face_hash, hash_flipped)
            
            # Tomar la mejor similitud
            sim = min(sim_normal, sim_flipped)
            
            if sim < best_similarity:
                best_similarity = sim
                best_match = name
                best_image_path = image_path
        
        # Retornar solo si está dentro del umbral
        if best_match and best_similarity <= threshold:
            # Convertir similitud a porcentaje (0-100)
            # Menos distancia = más similitud
            similarity_pct = max(0, 100 - (best_similarity / threshold * 100))
            return best_match, similarity_pct, best_image_path
        
        return None, None, None
    
    def _hamming_distance(self, hash1, hash2):
        """Calcula la distancia de Hamming entre dos hashes"""
        if len(hash1) != len(hash2):
            return float('inf')
        
        distance = 0
        for c1, c2 in zip(hash1, hash2):
            if c1 != c2:
                # Convertir hex a binario y contar bits diferentes
                val1 = int(c1, 16) if c1.isdigit() or c1.lower() in 'abcdef' else 0
                val2 = int(c2, 16) if c2.isdigit() or c2.lower() in 'abcdef' else 0
                xor = val1 ^ val2
                distance += bin(xor).count('1')
        
        return distance
    
    def get_all_persons(self):
        """Obtiene lista de todas las personas registradas"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, name, image_path FROM persons ORDER BY name')
        return cursor.fetchall()
    
    def delete_person(self, name):
        """Elimina una persona de la base de datos"""
        cursor = self.conn.cursor()
        
        # Obtener ruta de imagen
        cursor.execute('SELECT image_path FROM persons WHERE name = ?', (name,))
        result = cursor.fetchone()
        
        if result:
            image_path = result[0]
            
            # Eliminar imagen si existe
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except:
                    pass
            
            # Eliminar de BD
            cursor.execute('DELETE FROM persons WHERE name = ?', (name,))
            self.conn.commit()
            return True
        
        return False
    
    def update_person_image(self, name, new_face_image, hash_size=16):
        """Actualiza la imagen y hashes de una persona"""
        cursor = self.conn.cursor()
        
        # Verificar que existe
        cursor.execute('SELECT id, image_path FROM persons WHERE name = ?', (name,))
        result = cursor.fetchone()
        
        if not result:
            return False
        
        person_id, old_image_path = result
        
        # Eliminar imagen antigua
        if old_image_path and os.path.exists(old_image_path):
            try:
                os.remove(old_image_path)
            except:
                pass
        
        # Procesar nueva imagen
        if isinstance(new_face_image, np.ndarray):
            if len(new_face_image.shape) == 3 and new_face_image.shape[2] == 3:
                new_face_image = cv2.cvtColor(new_face_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(new_face_image)
        else:
            pil_image = new_face_image
        
        # Calcular nuevos hashes
        hash_normal = self._compute_hash(pil_image, hash_size)
        flipped = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
        hash_flipped = self._compute_hash(flipped, hash_size)
        
        # Guardar nueva imagen
        image_filename = f"{name.replace(' ', '_').lower()}.jpg"
        image_path = os.path.join(self.images_cache_dir, image_filename)
        pil_image.save(image_path, 'JPEG', quality=95)
        
        # Actualizar en BD
        cursor.execute('''
            UPDATE persons 
            SET hash_normal = ?, hash_flipped = ?, image_path = ?
            WHERE id = ?
        ''', (hash_normal, hash_flipped, image_path, person_id))
        
        self.conn.commit()
        return True
    
    def close(self):
        """Cierra la conexión a la base de datos"""
        if self.conn:
            self.conn.close()
            self.conn = None
