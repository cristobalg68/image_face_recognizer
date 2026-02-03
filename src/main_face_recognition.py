import os
import sys
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QFrame, QToolButton, QGridLayout, 
    QScrollArea, QPushButton, QMessageBox, QInputDialog,
    QDialog, QDialogButtonBox
)
from PyQt6.QtGui import QIcon, QPixmap, QImage
from PyQt6.QtCore import Qt, QSize, QTimer
import cv2
import numpy as np

from face_widget import FaceWidget
from video_frame import VideoFrame
from face_scanner import LiveFaceScanner
from dialogs import SelectFaceDialog

BASE = os.path.dirname(__file__)
ASSETS = os.path.join(BASE, "assets")


class FaceRecognitionWindow(QMainWindow):
    """Ventana principal para reconocimiento facial"""
    
    def __init__(self):
        super().__init__()
        
        # Configuración del scanner
        self.scanner = None
        self.settings = {
            "path_weights": "weights/yolov12l-face.pt",
            "size": 640,
            "confidence": 0.6,
            "iou": 0.5,
            "hash_size": 16,
            "db_path": "data/faces.db"
        }
        
        # Estados
        self.webcam_timer = None
        self.webcam_capture = None
        self.is_running = False
        
        # Personas detectadas (sin duplicados)
        self.detected_persons = {}  # {person_name: face_data}
        
        self.setWindowTitle("Sistema de Reconocimiento Facial")
        self.setMinimumSize(1320, 720)
        
        # Icono
        icon_path = os.path.join(ASSETS, "icon_app.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # Widget central
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)
        
        # Header
        header = QHBoxLayout()
        title = QLabel("Sistema de Reconocimiento Facial")
        title.setObjectName("appTitle")
        title.setStyleSheet("font-size: 20pt; font-weight: bold; color: #5865f2;")
        header.addWidget(title)
        header.addStretch()
        main_layout.addLayout(header)
        
        # Contenido principal
        content = QHBoxLayout()
        content.setSpacing(14)
        
        # ============= PANEL IZQUIERDO: VIDEO EN VIVO =============
        left_frame = QFrame()
        left_frame.setObjectName("leftSection")
        left_frame.setMinimumWidth(640)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setSpacing(16)
        left_layout.setContentsMargins(20, 20, 20, 20)
        
        # Título
        video_title = QLabel("Cámara en Vivo")
        video_title.setStyleSheet("font-size: 16pt; font-weight: bold;")
        video_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(video_title)
        
        # Video frame
        placeholder_path = os.path.join(ASSETS, "camera.svg")
        self.video_widget = VideoFrame(placeholder_path if os.path.exists(placeholder_path) else None)
        self.video_widget.setMinimumHeight(480)
        left_layout.addWidget(self.video_widget)
        
        # Controles
        controls = QHBoxLayout()
        controls.setSpacing(10)
        
        # Botón Start/Stop
        self.btn_start = QPushButton("▶ Iniciar Cámara")
        self.btn_start.setObjectName("startButton")
        self.btn_start.setMinimumHeight(45)
        self.btn_start.clicked.connect(self.toggle_camera)
        controls.addWidget(self.btn_start)
        
        # Botón para registrar persona
        self.btn_register = QPushButton("Registrar Persona")
        self.btn_register.setObjectName("registerButton")
        self.btn_register.setMinimumHeight(45)
        self.btn_register.setEnabled(False)
        self.btn_register.clicked.connect(self.register_person_dialog)
        controls.addWidget(self.btn_register)
        
        # Botón para limpiar detecciones
        self.btn_clear = QPushButton("Limpiar")
        self.btn_clear.setObjectName("clearButton")
        self.btn_clear.setMinimumHeight(45)
        self.btn_clear.clicked.connect(self.clear_detections)
        controls.addWidget(self.btn_clear)
        
        left_layout.addLayout(controls)
        
        content.addWidget(left_frame, 60)
        
        # ============= PANEL DERECHO: CARAS DETECTADAS =============
        right_frame = QFrame()
        right_frame.setObjectName("rightSection")
        right_layout = QVBoxLayout(right_frame)
        right_layout.setSpacing(16)
        right_layout.setContentsMargins(20, 20, 20, 20)
        
        # Título
        detection_title = QLabel("Personas Detectadas")
        detection_title.setStyleSheet("font-size: 16pt; font-weight: bold;")
        detection_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(detection_title)
        
        # Contador
        self.detection_counter = QLabel("0 persona(s) detectada(s)")
        self.detection_counter.setStyleSheet("font-size: 11pt; color: #888;")
        self.detection_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.detection_counter)
        
        # Scroll area para las caras
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        
        scroll_content = QWidget()
        self.grid_layout = QGridLayout(scroll_content)
        self.grid_layout.setSpacing(12)
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        scroll.setWidget(scroll_content)
        right_layout.addWidget(scroll)
        
        content.addWidget(right_frame, 40)
        
        main_layout.addLayout(content)
        
        # Aplicar estilos
        self.apply_styles()
    
    def apply_styles(self):
        """Aplica los estilos CSS a la ventana"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0b0d11;
            }
            
            QFrame#leftSection, QFrame#rightSection {
                background-color: #0f1116;
                border-radius: 12px;
                border: 1px solid rgba(255,255,255,0.08);
            }
            
            QPushButton {
                background-color: #5865f2;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 11pt;
                font-weight: bold;
            }
            
            QPushButton:hover {
                background-color: #4752c4;
            }
            
            QPushButton:pressed {
                background-color: #3c45a5;
            }
            
            QPushButton#registerButton {
                background-color: #43b581;
            }
            
            QPushButton#registerButton:hover {
                background-color: #3ca374;
            }
            
            QPushButton#clearButton {
                background-color: #ed4245;
            }
            
            QPushButton#clearButton:hover {
                background-color: #c03537;
            }
            
            QPushButton:disabled {
                background-color: #2c2f33;
                color: #666;
            }
        """)
    
    def toggle_camera(self):
        """Inicia o detiene la cámara"""
        if not self.is_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Inicia la cámara y el reconocimiento"""
        # Ya NO importamos aquí, LiveFaceScanner ya está importado al inicio
        
        # Crear scanner si no existe
        if not isinstance(self.scanner, LiveFaceScanner):
            try:
                print("Cargando modelo YOLO...")
                self.scanner = LiveFaceScanner(**self.settings)
                print("Scanner creado")
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Error al cargar el modelo:\n{str(e)}\n\n"
                    f"Asegúrate de tener el modelo YOLO en:\n{self.settings['path_weights']}"
                )
                return
        else:
            self.scanner.reset_tracker()
        
        # Abrir webcam
        self.webcam_capture = cv2.VideoCapture(0)
        
        if not self.webcam_capture.isOpened():
            QMessageBox.critical(self, "Error", "No se pudo abrir la cámara")
            return
        
        # Iniciar timer para actualizar frames
        self.webcam_timer = QTimer()
        self.webcam_timer.timeout.connect(self.update_frame)
        self.webcam_timer.start(33)  # ~30 FPS
        
        self.is_running = True
        self.btn_start.setText("⏸ Detener Cámara")
        self.btn_register.setEnabled(True)
        print("✅ Cámara iniciada")
    
    def stop_camera(self):
        """Detiene la cámara"""
        if self.webcam_timer:
            self.webcam_timer.stop()
            self.webcam_timer = None
        
        if self.webcam_capture:
            self.webcam_capture.release()
            self.webcam_capture = None
        
        self.is_running = False
        self.btn_start.setText("▶ Iniciar Cámara")
        self.btn_register.setEnabled(False)
        self.video_widget.clear_image()
        print("⏹ Cámara detenida")
    
    def update_frame(self):
        """Actualiza el frame de la webcam y procesa detecciones"""
        if not self.webcam_capture:
            return
        
        ret, frame = self.webcam_capture.read()
        
        if not ret:
            return
        
        # Procesar frame con el scanner
        if self.scanner:
            processed_frame = self.scanner.process_frame(frame)
            
            # Mostrar frame procesado
            self.video_widget.set_image(processed_frame)
            
            # Actualizar detecciones
            self.update_detections()
    
    def update_detections(self):
        """Actualiza el panel de detecciones con nuevas caras"""
        if not self.scanner:
            return
        
        # Obtener personas detectadas
        persons = self.scanner.get_detected_persons()
        
        # Verificar si hay nuevas personas
        for person_data in persons:
            person_name = person_data['name']
            is_unknown = person_data.get('is_unknown', False)
            
            # Para personas conocidas, usar el nombre como key
            # Para desconocidos, usar el nombre único (ej: "Desconocido #1")
            person_key = person_name
            
            if person_key not in self.detected_persons:
                # Nueva persona detectada
                self.detected_persons[person_key] = person_data
                self.add_face_to_grid(person_data)
            else:
                # Actualizar si tiene mejor similitud (para desconocidos similares)
                if is_unknown:
                    # Actualizar la imagen si es más reciente
                    self.detected_persons[person_key] = person_data
        
        # Actualizar contador
        self.detection_counter.setText(f"{len(self.detected_persons)} persona(s) detectada(s)")
    
    def add_face_to_grid(self, person_data):
        """Agrega una cara al grid de detecciones"""
        face_widget = FaceWidget(
            person_name=person_data['name'],
            similarity=person_data['similarity'],
            face_img=person_data.get('face_image'),
            db_img_path=person_data.get('db_image_path'),
            is_unknown=person_data.get('is_unknown', True)
        )
        
        count = self.grid_layout.count()
        row = count // 2  # 2 columnas
        col = count % 2
        
        self.grid_layout.addWidget(face_widget, row, col)
    
    def clear_detections(self):
        """Limpia todas las detecciones"""
        # Limpiar grid
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        # Resetear diccionario
        self.detected_persons = {}
        
        # Resetear tracker del scanner
        if self.scanner:
            self.scanner.reset_tracker()
        
        # Actualizar contador
        self.detection_counter.setText("0 persona(s) detectada(s)")
    
    def register_person_dialog(self):
        """Abre diálogo para registrar una persona desconocida"""
        # Buscar si hay alguna persona desconocida
        unknown_persons = [
            p for p in self.detected_persons.values() 
            if p.get('is_unknown', False)
        ]
        
        if not unknown_persons:
            QMessageBox.information(
                self,
                "Info",
                "No hay personas desconocidas para registrar"
            )
            return
        
        dialog = SelectFaceDialog(unknown_persons, self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_person = dialog.selected_person
            person_name = dialog.person_name
            face_image = selected_person.get('face_image')
            
            # Registrar en la base de datos
            try:
                success = self.scanner.register_new_person(person_name, face_image)
                
                if success:
                    QMessageBox.information(
                        self,
                        "Éxito",
                        f"¡Persona registrada exitosamente!\n\n{person_name}"
                    )
                    
                    # Limpiar detecciones para que se vuelva a detectar con el nuevo nombre
                    self.clear_detections()
                else:
                    QMessageBox.warning(
                        self,
                        "Advertencia",
                        f"La persona '{person_name}' ya está registrada"
                    )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Error al registrar persona:\n{str(e)}"
                )
    
    def closeEvent(self, event):
        """Cleanup al cerrar"""
        self.stop_camera()
        
        # Cerrar base de datos
        if self.scanner and hasattr(self.scanner, 'db'):
            self.scanner.db.close()
        
        event.accept()


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    # Mostrar mensaje de inicio
    print("\n" + "="*70)
    print("   SISTEMA DE RECONOCIMIENTO FACIAL")
    print("="*70)
    print()
    
    window = FaceRecognitionWindow()
    window.show()
    
    sys.exit(app.exec())
