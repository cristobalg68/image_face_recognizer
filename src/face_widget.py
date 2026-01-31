from PyQt6.QtWidgets import QFrame, QVBoxLayout, QLabel, QHBoxLayout
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
import cv2
import numpy as np


class FaceWidget(QFrame):
    """Widget para mostrar una cara detectada con su información"""
    
    def __init__(self, person_name="Desconocido", similarity=0.0, face_img=None, db_img_path=None, is_unknown=True):
        super().__init__()
        self.setObjectName("facePreview")
        self.setMinimumWidth(180)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        # Imagen de la cara
        img_lbl = QLabel()
        img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_lbl.setFixedSize(160, 160)
        img_lbl.setStyleSheet("background-color: #1a1d23; border-radius: 8px;")
        
        # Priorizar imagen de la base de datos si existe
        if db_img_path and not is_unknown:
            try:
                pix = QPixmap(db_img_path)
                if not pix.isNull():
                    pix = pix.scaled(
                        img_lbl.width(), 
                        img_lbl.height(), 
                        Qt.AspectRatioMode.KeepAspectRatio, 
                        Qt.TransformationMode.SmoothTransformation
                    )
                    img_lbl.setPixmap(pix)
            except:
                # Si falla, usar la imagen capturada
                if face_img is not None:
                    pix = self._convert_face_to_pixmap(face_img)
                    img_lbl.setPixmap(pix)
        elif face_img is not None:
            # Usar imagen capturada
            pix = self._convert_face_to_pixmap(face_img)
            img_lbl.setPixmap(pix)
        
        layout.addWidget(img_lbl, 0, Qt.AlignmentFlag.AlignHCenter)

        # Nombre y porcentaje de coincidencia
        if is_unknown:
            name_text = f"{person_name}"
            color = "#ffa500"  # Naranja para desconocidos
        else:
            name_text = f"{person_name}\nCoincidencia: {similarity:.1f}%"
            color = "#00ff00"  # Verde para conocidos
        
        name = QLabel(name_text)
        name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name.setStyleSheet(f"font-weight: 700; color: {color}; font-size: 11pt;")
        name.setWordWrap(True)
        layout.addWidget(name)
        
        # Estilo del frame
        if is_unknown:
            border_color = "#ffa500"
        else:
            border_color = "#00ff00"
        
        self.setStyleSheet(f"""
            QFrame#facePreview {{
                background-color: #0f1116;
                border: 2px solid {border_color};
                border-radius: 10px;
                padding: 5px;
            }}
        """)
    
    def _convert_face_to_pixmap(self, face_img):
        """
        Convierte una imagen de cara (numpy array) a QPixmap
        
        Args:
            face_img: numpy array en formato BGR
            
        Returns:
            QPixmap
        """
        if isinstance(face_img, np.ndarray):
            # Convertir BGR a RGB
            rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            
            q_img = QImage(
                rgb_img.data,
                w,
                h,
                bytes_per_line,
                QImage.Format.Format_RGB888
            )
            
            pixmap = QPixmap.fromImage(q_img)
            
            # Escalar manteniendo proporción
            pixmap = pixmap.scaled(
                160, 
                160, 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            
            return pixmap
        
        return QPixmap()
