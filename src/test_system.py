import sys


def check_import(module_name, package_name=None):
    """Verifica que un módulo pueda ser importado"""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✅ {package_name} - OK")
        return True
    except ImportError as e:
        print(f"❌ {package_name} - FALTA (pip install {package_name})")
        return False


def check_file_exists(filepath, description):
    """Verifica que un archivo exista"""
    import os
    if os.path.exists(filepath):
        print(f"✅ {description} - OK")
        return True
    else:
        print(f"⚠️  {description} - NO ENCONTRADO")
        print(f"   Ruta: {filepath}")
        return False


def main():
    print("=" * 70)
    print("   VERIFICACIÓN DEL SISTEMA DE RECONOCIMIENTO FACIAL")
    print("=" * 70)
    print()
    
    all_ok = True
    
    # Verificar módulos de Python
    print("1. Verificando dependencias de Python...")
    print("-" * 70)
    
    modules = [
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("PIL", "Pillow"),
        ("ultralytics", "ultralytics"),
        ("imagehash", "imagehash"),
        ("PyQt6", "PyQt6"),
    ]
    
    for module, package in modules:
        if not check_import(module, package):
            all_ok = False
    
    print()
    
    # Verificar archivos del sistema
    print("2. Verificando archivos del sistema...")
    print("-" * 70)
    
    files = [
        ("src/face_database.py", "Gestor de base de datos"),
        ("src/face_scanner.py", "Scanner de caras"),
        ("src/face_utils.py", "Utilidades"),
        ("src/face_widget.py", "Widget de cara"),
        ("src/video_frame.py", "Widget de video"),
        ("src/detector.py", "Detector YOLO"),
        ("src/main_face_recognition.py", "Aplicación principal"),
        ("src/init_database.py", "Inicializador de BD"),
    ]
    
    for filepath, desc in files:
        if not check_file_exists(filepath, desc):
            all_ok = False
    
    print()
    
    # Verificar modelo YOLO
    print("3. Verificando modelo YOLO...")
    print("-" * 70)
    
    model_paths = [
        "weights/yolov12n-face.pt",
        "weights/yolov12s-face.pt",
        "weights/yolov12m-face.pt",
        "weights/yolov12l-face.pt",
    ]
    
    model_found = False
    for model_path in model_paths:
        if check_file_exists(model_path, f"Modelo YOLO ({model_path})"):
            model_found = True
            break
    
    if not model_found:
        print()
        print("⚠️  ADVERTENCIA: No se encontró ningún modelo YOLO")
        print("   Descarga un modelo de detección facial YOLO y colócalo en:")
        print("   weights/yolov8n-face.pt")
        print()
        print("   Opciones:")
        print("   - https://github.com/akanametov/yolov8-face")
        print("   - https://github.com/ultralytics/ultralytics")
    
    print()
    
    # Verificar directorios
    print("4. Verificando estructura de directorios...")
    print("-" * 70)
    
    import os
    
    dirs = [
        ("data", "Directorio de datos"),
        ("weights", "Directorio de modelos"),
    ]
    
    for dirpath, desc in dirs:
        if not os.path.exists(dirpath):
            print(f"⚠️  {desc} - CREANDO...")
            os.makedirs(dirpath, exist_ok=True)
            print(f"✅ {desc} - CREADO")
        else:
            print(f"✅ {desc} - OK")
    
    print()
    
    # Test rápido de la cámara
    print("5. Verificando acceso a la cámara...")
    print("-" * 70)
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                print(f"✅ Cámara - OK (resolución: {frame.shape[1]}x{frame.shape[0]})")
            else:
                print("⚠️  Cámara - Conectada pero sin imagen")
                all_ok = False
        else:
            print("❌ Cámara - NO DISPONIBLE")
            print("   Verifica que:")
            print("   - Tu cámara esté conectada")
            print("   - No esté siendo usada por otra aplicación")
            print("   - Tengas permisos para acceder a la cámara")
            all_ok = False
    except Exception as e:
        print(f"❌ Error al verificar cámara: {e}")
        all_ok = False
    
    print()
    
    # Test de base de datos
    print("6. Verificando base de datos...")
    print("-" * 70)
    
    try:
        from face_database import FaceDatabase
        
        db = FaceDatabase("data/test_faces.db", "data/test_face_images")
        db.connect()
        
        # Probar operaciones básicas
        print("✅ Conexión a BD - OK")
        
        # Limpiar test
        import os
        db.close()
        
        if os.path.exists("data/test_faces.db"):
            os.remove("data/test_faces.db")
        
        if os.path.exists("data/test_face_images") and len(os.listdir("data/test_face_images")) == 0:
            os.rmdir("data/test_face_images")
        
        print("✅ Operaciones de BD - OK")
        
    except Exception as e:
        print(f"❌ Error en base de datos: {e}")
        all_ok = False
    
    print()
    print("=" * 70)
    
    if all_ok and model_found:
        print("✅ SISTEMA LISTO PARA USAR")
        print()
        print("Siguiente paso:")
        print("  1. Ejecuta: python init_database.py")
        print("     (Para registrar personas en la base de datos)")
        print()
        print("  2. Ejecuta: python main_face_recognition.py")
        print("     (Para iniciar el reconocimiento facial)")
    elif all_ok and not model_found:
        print("⚠️  CASI LISTO - FALTA EL MODELO YOLO")
        print()
        print("Descarga un modelo YOLO de detección facial y colócalo en:")
        print("  weights/yolov8n-face.pt")
    else:
        print("❌ HAY PROBLEMAS QUE RESOLVER")
        print()
        print("Revisa los errores arriba e instala las dependencias faltantes:")
        print("  pip install -r requirements.txt")
    
    print("=" * 70)
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
