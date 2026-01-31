"""
Script de diagnóstico para problemas con PyTorch en Windows
"""

import sys
import os

print("="*70)
print("   DIAGNÓSTICO DE PyTorch")
print("="*70)
print()

# 1. Información del sistema
print("1. INFORMACIÓN DEL SISTEMA")
print("-"*70)
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Platform: {sys.platform}")
print()

# 2. Verificar instalación de torch
print("2. VERIFICAR INSTALACIÓN DE PyTorch")
print("-"*70)

try:
    import torch
    print("✅ PyTorch se importó correctamente")
    print(f"   Versión: {torch.__version__}")
    print(f"   CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
    print()
    
except ImportError as e:
    print(f"❌ Error al importar PyTorch: {e}")
    print("\nSOLUCIÓN:")
    print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)
    
except OSError as e:
    print(f"❌ Error de DLL al importar PyTorch: {e}")
    print("\n" + "="*70)
    print("   PROBLEMA DETECTADO: Faltan DLLs de Visual C++")
    print("="*70)
    print("\nSOLUCIONES EN ORDEN DE PREFERENCIA:")
    print("\n1. INSTALAR Visual C++ Redistributable:")
    print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("\n2. REINSTALAR PyTorch (versión CPU):")
    print("   pip uninstall torch torchvision torchaudio -y")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
    print("\n3. USAR CONDA (si tienes Anaconda):")
    print("   conda install pytorch torchvision cpuonly -c pytorch")
    print()
    sys.exit(1)

# 3. Verificar ultralytics
print("3. VERIFICAR ULTRALYTICS")
print("-"*70)

try:
    from ultralytics import YOLO
    print("✅ Ultralytics se importó correctamente")
    
    # Intentar cargar un modelo
    print("   Probando carga de modelo...")
    try:
        model = YOLO('yolov8n.pt')  # Se descarga automáticamente si no existe
        print("✅ Modelo YOLO cargado correctamente")
    except Exception as e:
        print(f"⚠️  Error al cargar modelo: {e}")
        
except ImportError as e:
    print(f"❌ Error al importar Ultralytics: {e}")
    print("\nSOLUCIÓN:")
    print("pip install ultralytics")

print()

# 4. Verificar otras dependencias
print("4. VERIFICAR OTRAS DEPENDENCIAS")
print("-"*70)

dependencies = [
    ('cv2', 'opencv-python'),
    ('PIL', 'Pillow'),
    ('imagehash', 'imagehash'),
    ('PyQt6', 'PyQt6'),
    ('numpy', 'numpy'),
]

all_ok = True
for module, package in dependencies:
    try:
        __import__(module)
        print(f"✅ {package}")
    except ImportError:
        print(f"❌ {package} - FALTA")
        print(f"   Instalar con: pip install {package}")
        all_ok = False

print()

# 5. Resumen
print("="*70)
if all_ok:
    print("✅ TODAS LAS DEPENDENCIAS INSTALADAS CORRECTAMENTE")
    print("\nPuedes ejecutar:")
    print("  python src/main_face_recognition.py")
else:
    print("⚠️  HAY DEPENDENCIAS FALTANTES")
    print("\nInstala las faltantes con:")
    print("  pip install -r requirements.txt")

print("="*70)
print()
