# 🎥 Sistema Avanzado de Reconocimiento Facial

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyQt6](https://img.shields.io/badge/PyQt6-6.5%2B-green)
![YOLO](https://img.shields.io/badge/YOLO-v8-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

Sistema de reconocimiento facial en tiempo real con detección mejorada, registro inteligente y base de datos SQLite.

[Características](#-características) • [Instalación](#-instalación) • [Uso](#-uso) • [Configuración](#%EF%B8%8F-configuración) • [API](#-api)

</div>

---

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
  - [Instalación Rápida](#instalación-rápida)
  - [Instalación Detallada](#instalación-detallada)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Uso](#-uso)
  - [Inicio Rápido](#inicio-rápido)
  - [Registrar Personas](#registrar-personas)
  - [Gestión de Base de Datos](#gestión-de-base-de-datos)
- [Configuración](#%EF%B8%8F-configuración)
  - [Parámetros del Detector](#parámetros-del-detector)
  - [Ajuste de Reconocimiento](#ajuste-de-reconocimiento)
- [Arquitectura](#-arquitectura)
- [API y Módulos](#-api-y-módulos)
- [Solución de Problemas](#-solución-de-problemas)
- [Mejoras Futuras](#-mejoras-futuras)
- [Contribuir](#-contribuir)
- [Licencia](#-licencia)

---

## ✨ Características

### 🎯 Detección Mejorada
- **Filtrado Inteligente**: Reduce falsos positivos mediante múltiples criterios
  - Validación de confianza mínima
  - Verificación de tamaño de cara
  - Análisis de aspect ratio
  - Detección de tonos de piel
- **Resistente a Cambios**: Funciona con diferentes:
  - Condiciones de iluminación
  - Poses y ángulos de la cara
  - Expresiones faciales
  - Calidad de imagen

### 👤 Reconocimiento Robusto
- **Hashing Múltiple**: Combina dhash, phash y ahash para mayor precisión
- **Normalización de Imágenes**: Ecualización adaptativa de histograma
- **Variaciones de Búsqueda**: Prueba múltiples transformaciones:
  - Imagen normal
  - Imagen espejo
  - Rotaciones leves (±5°)
- **Tracking Inteligente**: Mantiene IDs consistentes entre frames

### 💾 Base de Datos SQLite
- **Almacenamiento Eficiente**: Hashes perceptuales indexados
- **Búsqueda Rápida**: Índices optimizados para consultas
- **Gestión Completa**: CRUD de personas registradas
- **Portabilidad**: Base de datos en un solo archivo

### 🖥️ Interfaz Moderna
- **Diseño Intuitivo**: GUI profesional con PyQt6
- **Visualización en Tiempo Real**: Video con detecciones superpuestas
- **Panel de Detecciones**: Grid con todas las personas detectadas
- **Selección de Caras**: Diálogo para elegir qué cara registrar
- **Estadísticas**: Contador de personas y métricas de filtrado

### 🔐 Privacidad
- **100% Local**: Sin conexión a internet
- **Datos Privados**: Todo se almacena localmente
- **Control Total**: Elimina personas cuando quieras

---

## 📦 Requisitos

### Sistema Operativo
- Windows 10/11
- Linux (Ubuntu 20.04+)
- macOS 10.15+

### Software
- **Python**: 3.8 o superior
- **Webcam**: Cámara integrada o USB

### Hardware Recomendado
- **CPU**: Intel i5 / AMD Ryzen 5 o superior
- **RAM**: 4GB mínimo, 8GB recomendado
- **GPU**: Opcional (CPU es suficiente con PyTorch CPU)

---

## 🚀 Instalación

### Instalación Rápida

```bash
# 1. Clonar o descargar el proyecto
cd face_recognition_system

# 2. Crear entorno virtual (recomendado)
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar modelo YOLO
# Opción A: Modelo especializado en caras (recomendado)
# Descarga de: https://github.com/akanametov/yolov12l-face
# Coloca en: weights/yolov12l-face.pt

# Opción B: Modelo general 
python -c "from ultralytics import YOLO; YOLO('yolov12l.pt')"
# Mueve el archivo descargado a: weights/yolov12l.pt

# 6. ¡Ejecutar!
python main_face_recognition.py
```

### Instalación Detallada

<details>
<summary><b>📖 Ver pasos detallados</b></summary>

#### 1. Preparar el Entorno

```bash
# Verificar Python
python --version  # Debe ser 3.8+

# Crear directorio del proyecto
mkdir face_recognition_system
cd face_recognition_system

# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Windows CMD:
.\.venv\Scripts\activate.bat

# Linux/Mac:
source .venv/bin/activate
```

#### 2. Instalar Dependencias

```bash
# Instalar paquetes base
pip install --upgrade pip

# Opción A: Desde requirements.txt
pip install -r requirements.txt

# Opción B: Manual
pip install opencv-python>=4.8.0
pip install PyQt6>=6.5.0
pip install ultralytics>=8.0.0
pip install imagehash>=4.3.1
pip install Pillow>=10.0.0
pip install numpy>=1.24.0

# PyTorch (versión CPU, más liviana)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### 3. Obtener Modelo YOLO

**Opción A - Modelo Especializado en Caras (Recomendado):**

1. Visita: https://github.com/akanametov/yolov12-face
2. Descarga el modelo `yolov12l-face.pt`
3. Colócalo en: `weights/yolov12l-face.pt`

**Opción B - Entrenar tu Propio Modelo:**

```python
from ultralytics import YOLO

# Cargar modelo base
model = YOLO('yolov12l.pt')

# Entrenar con tu dataset
model.train(
    data='faces.yaml',  # Tu dataset de caras
    epochs=100,
    imgsz=640,
    batch=16
)
```

</details>

---

## 📁 Estructura del Proyecto

```
face_recognition_system/
│
├── data/                              # Datos de la aplicación
│   ├── faces.db                       # Base de datos SQLite
│   ├── face_images/                   # Caché de imágenes de caras
│   └── training_faces/                # Carpeta para registro inicial
│
├── weights/                           # Modelos YOLO
│   └── yolov12l-face.pt               # Modelo de detección facial
│
├── src/                              # Código fuente principal
│   ├── detector.py                   # Detector YOLO
│   ├── face_database.py              # Gestor de base de datos
│   ├── face_scanner.py               # Scanners (Live/Video/Image)
│   ├── face_utils.py                 # Utilidades
│   ├── face_widget.py                # Widget de cara detectada
│   ├── video_frame.py                # Widget de visualización
│   ├── dialogs.py                    # Diálogos de interfaz
|   ├── init_database.py              # Inicializar BD
│   └── main_face_recognition.py      # Aplicación principal
│
├── requirements.txt                 # Dependencias Python
├── README.md                        # Este archivo
└── .gitignore                       # Archivos ignorados por Git
```

---

## 🎮 Uso

### Inicio Rápido

```bash
# 1. Activar entorno virtual
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\activate   # Windows

# 2. Ejecutar aplicación
python src/main_face_recognition.py

# 3. En la interfaz:
#    - Click "Iniciar Cámara"
#    - Mira cómo detecta caras automáticamente
#    - Las conocidas aparecen en verde
#    - Las desconocidas en naranja
```

### Registrar Personas

#### Método 1: Desde la Interfaz (Recomendado)

1. **Iniciar la cámara**
2. **Mostrar cara desconocida** ante la cámara
3. **Click en "Registrar Persona"**
4. **Seleccionar la cara** que deseas registrar
5. **Ingresar el nombre** de la persona
6. **Click en "Registrar"**

La persona ahora será reconocida automáticamente.

#### Método 2: Registro por Lote

```bash
# 1. Crear directorio de entrenamiento
mkdir -p data/training_faces

# 2. Agregar fotos de caras
# Nombra las fotos con el nombre de la persona:
#   juan_perez.jpg
#   maria_garcia.jpg
#   carlos_lopez.jpg

# 3. Ejecutar script de inicialización
python scripts/init_database.py

# 4. Seleccionar opción 1 para registro automático
```

#### Método 3: Mediante Código

```python
from face_database import FaceDatabase
import cv2

# Conectar a la BD
db = FaceDatabase("data/faces.db")
db.connect()

# Cargar imagen
img = cv2.imread("foto_persona.jpg")

# Registrar persona
success = db.register_person("Juan Pérez", img, hash_size=16)

if success:
    print("Persona registrada exitosamente")
else:
    print("La persona ya existe en la BD")

# Cerrar conexión
db.close()
```

### Gestión de Base de Datos

#### Listar Personas Registradas

```python
from face_database import FaceDatabase

db = FaceDatabase("data/faces.db")
db.connect()

# Obtener todas las personas
persons = db.get_all_persons()

for person_id, name, img_path in persons:
    print(f"ID: {person_id} | Nombre: {name}")
    print(f"Imagen: {img_path}\n")

db.close()
```

#### Eliminar Persona

```python
from face_database import FaceDatabase

db = FaceDatabase("data/faces.db")
db.connect()

# Eliminar por nombre
success = db.delete_person("Juan Pérez")

if success:
    print("Persona eliminada")
else:
    print("Persona no encontrada")

db.close()
```

#### Actualizar Imagen de Persona

```python
from face_database import FaceDatabase
import cv2

db = FaceDatabase("data/faces.db")
db.connect()

# Cargar nueva imagen
new_img = cv2.imread("nueva_foto.jpg")

# Actualizar
success = db.update_person_image("Juan Pérez", new_img)

db.close()
```

---

## ⚙️ Configuración

### Parámetros del Detector

Edita `src/detector_improved.py`:

```python
detector = ImprovedDetector(
    path_weights="weights/yolov8n-face.pt",
    min_confidence=0.6,        # Confianza mínima (0-1)
    min_face_size=40,          # Tamaño mínimo en píxeles
    max_aspect_ratio=2.5       # Ratio máximo ancho/alto
)
```

**Recomendaciones:**

| Escenario | min_confidence | min_face_size | max_aspect_ratio |
|-----------|---------------|---------------|------------------|
| Ambiente controlado | 0.7 | 60 | 2.0 |
| Uso general | 0.6 | 40 | 2.5 |
| Muchas caras | 0.5 | 30 | 3.0 |

### Ajuste de Reconocimiento

Edita `src/main_face_recognition.py`:

```python
self.settings = {
    "path_weights": "weights/yolov12l-face.pt",
    "size": 640,              # Tamaño de procesamiento
    "confidence": 0.6,        # Umbral de confianza
    "iou": 0.5,              # Umbral de IoU
    "hash_size": 16,         # Tamaño del hash
    "db_path": "data/faces.db"
}
```

**Valores Recomendados:**

#### Tamaño de Procesamiento (`size`)
- `416`: Más rápido, menos preciso
- `640`: **Balanceado (recomendado)**
- `1280`: Más preciso, más lento

#### Confianza (`confidence`)
- `0.4-0.5`: Más permisivo, más detecciones
- `0.6-0.7`: **Balanceado (recomendado)**
- `0.8-0.9`: Muy estricto, menos falsos positivos

#### Hash Size (`hash_size`)
- `8`: Más rápido, menos preciso
- `16`: **Balanceado (recomendado)**
- `32`: Más preciso, más lento

### Umbral de Similitud

Edita `src/face_database.py`:

```python
def find_match(self, face_hash, threshold=150.0):
    # threshold: Distancia Hamming máxima para match
    # Valores más bajos = más estricto
```

**Recomendaciones:**

| threshold | Comportamiento |
|-----------|---------------|
| 100-120 | Muy estricto (gemelos se distinguen) |
| **130-150** | **Balanceado (recomendado)** |
| 160-180 | Permisivo (puede agrupar personas similares) |

### Similitud entre Desconocidos

Edita `src/face_utils_improved.py` donde se llama `track_faces`:

```python
utils.track_faces(
    detections, 
    tracker, 
    iou_threshold=0.5,
    unknown_similarity_threshold=80.0  # Ajusta aquí
)
```

**Valores:**
- `60-70`: Más estricto (más desconocidos diferentes)
- `80-90`: **Balanceado (recomendado)**
- `100-120`: Permisivo (agrupa desconocidos similares)

---

## 🏗️ Arquitectura

### Flujo de Procesamiento

```
┌─────────────────────────────────────────────────────────────┐
│                     ENTRADA: Frame de Video                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              1. DETECCIÓN (detector_improved.py)             │
│  • YOLO detecta caras                                        │
│  • Filtros de validación:                                    │
│    - Confianza mínima                                        │
│    - Tamaño mínimo                                          │
│    - Aspect ratio                                           │
│    - Detección de tonos de piel                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│          2. EXTRACCIÓN (face_utils_improved.py)              │
│  • Extraer región de cada cara                               │
│  • Agregar padding                                          │
│  • Normalizar iluminación (CLAHE)                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            3. HASHING (face_utils_improved.py)               │
│  • Calcular múltiples hashes:                                │
│    - dhash (diferencias)                                     │
│    - phash (perceptual)                                     │
│    - ahash (average)                                        │
│  • Variaciones:                                             │
│    - Normal                                                 │
│    - Espejo                                                 │
│    - Rotación +5°                                           │
│    - Rotación -5°                                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│          4. BÚSQUEDA (face_database.py)                      │
│  • Comparar hashes con base de datos                         │
│  • Calcular distancia Hamming                               │
│  • Probar todas las variaciones                             │
│  • Seleccionar mejor match                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│             5. TRACKING (face_utils_improved.py)             │
│  • Asociar caras entre frames (IoU)                         │
│  • Mantener IDs consistentes                                │
│  • Agrupar desconocidos similares                           │
│  • Asignar IDs únicos a nuevos desconocidos                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│       6. VISUALIZACIÓN (main_face_recognition.py)            │
│  • Dibujar rectángulos en video                             │
│  • Mostrar nombres y confianza                              │
│  • Actualizar panel de detecciones                          │
│  • Registrar nuevas personas                                │
└─────────────────────────────────────────────────────────────┘
```

### Diagrama de Clases

```
┌──────────────────────┐
│  ImprovedDetector    │
│──────────────────────│
│ + detect_objects()   │
│ + filter_detections()│
│ + has_skin_tone()    │
│ + get_stats()        │
└──────────────────────┘
           │
           │ usa
           ▼
┌──────────────────────┐
│   FaceDatabase       │
│──────────────────────│
│ + register_person()  │
│ + find_match()       │
│ + get_all_persons()  │
│ + delete_person()    │
│ + update_image()     │
└──────────────────────┘
           │
           │ usa
           ▼
┌──────────────────────┐
│  LiveFaceScanner     │
│──────────────────────│
│ + process_frame()    │
│ + reset_tracker()    │
│ + get_detected()     │
│ + register_new()     │
└──────────────────────┘
           │
           │ usa
           ▼
┌──────────────────────┐
│   FaceRecognition    │
│   Window (GUI)       │
│──────────────────────│
│ + start_camera()     │
│ + update_frame()     │
│ + update_detections()│
│ + register_dialog()  │
└──────────────────────┘
```

---

## 🚧 Mejoras Futuras

### Roadmap
- [ ] Soporte para múltiples cámaras simultáneas
- [ ] Modo de video pregrabado con procesamiento por lotes
- [ ] Exportar detecciones a CSV/Excel
- [ ] Historial de detecciones con timestamps
- [ ] Reconocimiento de emociones
- [ ] Estimación de edad y género
- [ ] Integración con sistemas de control de acceso
- [ ] Alertas configurables (email/SMS cuando se detecta persona específica)
- [ ] API REST para integración con otros sistemas
- [ ] App móvil (iOS/Android)
- [ ] Dashboard web para visualización de estadísticas

---

<div align="center">

**⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub ⭐**

</div>
