"""
Diálogo mejorado para seleccionar y registrar caras desconocidas
Permite elegir cuál cara desconocida registrar
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QWidget, QFrame, QLineEdit, QMessageBox,
    QRadioButton, QButtonGroup
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
import cv2
import numpy as np


class SelectFaceDialog(QDialog):
    """Diálogo para seleccionar qué cara desconocida registrar"""
    
    def __init__(self, unknown_persons, parent=None):
        super().__init__(parent)
        
        self.unknown_persons = unknown_persons  # Lista de datos de caras desconocidas
        self.selected_person = None
        self.person_name = None
        
        self.setWindowTitle("Seleccionar Cara para Registrar")
        self.setModal(True)
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Título
        title = QLabel("Selecciona la cara que deseas registrar:")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; color: white;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Contador
        count_label = QLabel(f"{len(unknown_persons)} cara(s) desconocida(s) detectada(s)")
        count_label.setStyleSheet("font-size: 11pt; color: #888;")
        count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(count_label)
        
        # Scroll area para las caras
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #3a3f4b;
                border-radius: 8px;
                background-color: #1a1d23;
            }
        """)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(10)
        scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Grupo de radio buttons para selección única
        self.button_group = QButtonGroup(self)
        
        # Agregar cada cara desconocida
        for i, person_data in enumerate(unknown_persons):
            face_item = self._create_face_item(i, person_data)
            scroll_layout.addWidget(face_item)
        
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        # Input para el nombre
        name_layout = QVBoxLayout()
        name_layout.setSpacing(8)
        
        name_instruction = QLabel("Nombre de la persona:")
        name_instruction.setStyleSheet("font-size: 11pt; color: white;")
        name_layout.addWidget(name_instruction)
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Ingresa el nombre completo...")
        self.name_input.setStyleSheet("""
            QLineEdit {
                padding: 10px;
                font-size: 11pt;
                border: 2px solid #3a3f4b;
                border-radius: 5px;
                background-color: #1a1d23;
                color: white;
            }
            QLineEdit:focus {
                border-color: #5865f2;
            }
        """)
        name_layout.addWidget(self.name_input)
        
        layout.addLayout(name_layout)
        
        # Botones
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)
        
        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.setObjectName("cancelButton")
        self.btn_cancel.setMinimumHeight(45)
        self.btn_cancel.clicked.connect(self.reject)
        buttons_layout.addWidget(self.btn_cancel)
        
        self.btn_register = QPushButton("Registrar")
        self.btn_register.setObjectName("registerButton")
        self.btn_register.setMinimumHeight(45)
        self.btn_register.clicked.connect(self.register_person)
        buttons_layout.addWidget(self.btn_register)
        
        layout.addLayout(buttons_layout)
        
        # Estilos
        self.setStyleSheet("""
            QDialog {
                background-color: #0f1116;
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
            
            QPushButton#cancelButton {
                background-color: #ed4245;
            }
            
            QPushButton#cancelButton:hover {
                background-color: #c03537;
            }
            
            QPushButton#registerButton {
                background-color: #43b581;
            }
            
            QPushButton#registerButton:hover {
                background-color: #3ca374;
            }
        """)
    
    def _create_face_item(self, index, person_data):
        """Crea un widget para mostrar una cara"""
        frame = QFrame()
        frame.setObjectName("faceItem")
        frame.setStyleSheet("""
            QFrame#faceItem {
                background-color: #1a1d23;
                border: 2px solid #3a3f4b;
                border-radius: 8px;
                padding: 10px;
            }
            QFrame#faceItem:hover {
                border-color: #5865f2;
            }
        """)
        
        layout = QHBoxLayout(frame)
        layout.setSpacing(15)
        
        # Radio button para selección
        radio = QRadioButton()
        radio.setStyleSheet("""
            QRadioButton {
                color: white;
                font-size: 11pt;
            }
            QRadioButton::indicator {
                width: 20px;
                height: 20px;
            }
        """)
        self.button_group.addButton(radio, index)
        
        # Seleccionar el primero por defecto
        if index == 0:
            radio.setChecked(True)
        
        layout.addWidget(radio)
        
        # Imagen de la cara
        face_img = person_data.get('face_image')
        
        img_label = QLabel()
        img_label.setFixedSize(100, 100)
        img_label.setStyleSheet("border: 1px solid #3a3f4b; border-radius: 5px;")
        
        if face_img is not None:
            pixmap = self._convert_face_to_pixmap(face_img)
            img_label.setPixmap(pixmap)
        
        layout.addWidget(img_label)
        
        # Información
        info_layout = QVBoxLayout()
        info_layout.setSpacing(5)
        
        name_label = QLabel(person_data['name'])
        name_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: #ffa500;")
        info_layout.addWidget(name_label)
        
        # Mostrar confianza de detección si está disponible
        confidence = person_data.get('similarity', 0)
        if 'confidence' in person_data:
            confidence = person_data['confidence']
        
        conf_label = QLabel(f"Confianza: {confidence:.1f}%")
        conf_label.setStyleSheet("font-size: 10pt; color: #888;")
        info_layout.addWidget(conf_label)
        
        info_layout.addStretch()
        
        layout.addLayout(info_layout)
        layout.addStretch()
        
        return frame
    
    def _convert_face_to_pixmap(self, face_img):
        """Convierte imagen de cara a QPixmap"""
        if isinstance(face_img, np.ndarray):
            rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            return pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, 
                                Qt.TransformationMode.SmoothTransformation)
        return QPixmap()
    
    def register_person(self):
        """Valida y registra la persona seleccionada"""
        # Validar que haya un nombre
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Por favor ingrese un nombre")
            return
        
        # Obtener la cara seleccionada
        selected_id = self.button_group.checkedId()
        if selected_id < 0:
            QMessageBox.warning(self, "Error", "Por favor seleccione una cara")
            return
        
        # Guardar datos
        self.selected_person = self.unknown_persons[selected_id]
        self.person_name = name
        
        self.accept()