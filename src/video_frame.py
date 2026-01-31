from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QPixmap, QImage, QPainter
from PyQt6.QtCore import Qt, QSize
import numpy as np

class VideoFrame(QLabel):
    def __init__(self, placeholder_path=None):
        super().__init__()
        self.setObjectName("videoFrame")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(False)
        
        self.setStyleSheet("""
            QLabel#videoFrame {
                background-color: #0b0d11;
                border-radius: 10px;
                border: 1px solid rgba(255,255,255,0.08);
            }
        """)

        self.original_pixmap = None
        self.placeholder_pixmap = None

        if placeholder_path:
            pix = QPixmap(placeholder_path)
            self.placeholder_pixmap = pix
            self.original_pixmap = pix

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def _update_display(self):
        if self.original_pixmap and not self.original_pixmap.isNull():
            # Calcular el tamaño escalado manteniendo aspect ratio
            scaled = self.original_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            super().setPixmap(scaled)
        elif self.placeholder_pixmap:
            # Para el placeholder, usar un tamaño más pequeño y centrado
            target_size = QSize(
                min(self.width() - 40, 120),
                min(self.height() - 40, 120)
            )
            scaled = self.placeholder_pixmap.scaled(
                target_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            super().setPixmap(scaled)

    def set_image(self, image):
        """Establece una nueva imagen en el frame"""
        if isinstance(image, str):
            pix = QPixmap(image)
        elif isinstance(image, QPixmap):
            pix = image
        elif isinstance(image, QImage):
            pix = QPixmap.fromImage(image)
        elif isinstance(image, np.ndarray):
            pix = self._ndarray_to_pixmap(image)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        self.original_pixmap = pix
        self._update_display()

    def _ndarray_to_pixmap(self, img: np.ndarray) -> QPixmap:
        """Convierte numpy/OpenCV image → QPixmap (PyQt6 safe)"""
        if img.ndim == 2:
            # Grayscale
            h, w = img.shape
            qimg = QImage(
                img.tobytes(),
                w,
                h,
                w,
                QImage.Format.Format_Grayscale8
            )
        elif img.ndim == 3:
            h, w, ch = img.shape
            if ch == 3:
                # BGR → RGB
                img = img[:, :, ::-1]
                fmt = QImage.Format.Format_RGB888
                bytes_per_line = 3 * w
            elif ch == 4:
                # BGRA → RGBA
                img = img[:, :, [2, 1, 0, 3]]
                fmt = QImage.Format.Format_RGBA8888
                bytes_per_line = 4 * w
            else:
                raise ValueError("Formato de imagen no soportado")

            qimg = QImage(
                img.tobytes(),
                w,
                h,
                bytes_per_line,
                fmt
            )
        else:
            raise ValueError("Dimensiones de imagen no soportadas")

        return QPixmap.fromImage(qimg)

    def clear_image(self):
        """Limpia la imagen y muestra el placeholder"""
        self.original_pixmap = self.placeholder_pixmap
        self._update_display()

    def has_content(self):
        """Retorna True si hay contenido diferente al placeholder"""
        return (self.original_pixmap is not None and 
                self.original_pixmap != self.placeholder_pixmap)