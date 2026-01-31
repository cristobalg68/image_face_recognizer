"""
Script de ejemplo para probar reconocimiento facial sin GUI
Útil para debugging y pruebas rápidas
"""

import cv2
import os
from face_database import FaceDatabase
from face_scanner import LiveFaceScanner


def test_recognition(duration_seconds=10):
    """
    Prueba el reconocimiento facial durante X segundos
    
    Args:
        duration_seconds: Duración de la prueba en segundos
    """
    print("=" * 70)
    print("   TEST DE RECONOCIMIENTO FACIAL")
    print("=" * 70)
    print()
    
    # Configuración
    settings = {
        "path_weights": "weights/yolov8n-face.pt",
        "size": 640,
        "confidence": 0.5,
        "iou": 0.5,
        "hash_size": 16,
        "db_path": "data/faces.db"
    }
    
    # Verificar que existe la base de datos
    if not os.path.exists(settings["db_path"]):
        print("⚠️  ADVERTENCIA: No existe la base de datos")
        print(f"   Ejecuta primero: python init_database.py")
        print()
        return
    
    # Verificar modelo
    if not os.path.exists(settings["path_weights"]):
        print(f"❌ ERROR: No se encontró el modelo YOLO en:")
        print(f"   {settings['path_weights']}")
        print()
        return
    
    print("✅ Configuración cargada")
    print(f"   Modelo: {settings['path_weights']}")
    print(f"   Base de datos: {settings['db_path']}")
    print()
    
    # Crear scanner
    try:
        print("⏳ Cargando modelo YOLO...")
        scanner = LiveFaceScanner(**settings)
        print("✅ Modelo cargado")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return
    
    # Verificar personas registradas
    persons = scanner.db.get_all_persons()
    print(f"\n📋 Personas registradas: {len(persons)}")
    
    if persons:
        for person_id, name, img_path in persons:
            print(f"   • {name}")
    else:
        print("   ⚠️  No hay personas registradas")
        print("   Ejecuta: python init_database.py")
    
    print()
    
    # Abrir cámara
    print("📹 Abriendo cámara...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: No se pudo abrir la cámara")
        return
    
    print("✅ Cámara abierta")
    print()
    print(f"⏱️  Iniciando reconocimiento por {duration_seconds} segundos...")
    print("   Presiona 'q' para salir antes")
    print()
    
    # Variables de control
    import time
    start_time = time.time()
    frame_count = 0
    detected_persons_set = set()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("⚠️  No se pudo leer el frame")
                break
            
            # Procesar frame
            processed_frame = scanner.process_frame(frame)
            
            # Obtener detecciones
            persons_detected = scanner.get_detected_persons()
            
            # Actualizar conjunto de personas detectadas
            for person_data in persons_detected:
                person_name = person_data['name']
                similarity = person_data['similarity']
                is_unknown = person_data.get('is_unknown', False)
                
                if person_name not in detected_persons_set:
                    detected_persons_set.add(person_name)
                    
                    if is_unknown:
                        print(f"🔍 Nueva cara detectada: {person_name}")
                    else:
                        print(f"✅ Persona reconocida: {person_name} ({similarity:.1f}%)")
            
            # Mostrar frame
            cv2.imshow('Reconocimiento Facial - Test', processed_frame)
            
            frame_count += 1
            
            # Verificar tecla de salida
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⏹️  Detenido por el usuario")
                break
            
            # Verificar tiempo
            elapsed = time.time() - start_time
            if elapsed >= duration_seconds:
                print(f"\n⏱️  Tiempo completado: {duration_seconds}s")
                break
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrumpido por el usuario")
    
    finally:
        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()
        scanner.db.close()
    
    # Mostrar resumen
    print()
    print("=" * 70)
    print("   RESUMEN")
    print("=" * 70)
    print(f"Frames procesados: {frame_count}")
    print(f"Tiempo total: {time.time() - start_time:.1f}s")
    print(f"FPS promedio: {frame_count / (time.time() - start_time):.1f}")
    print(f"Personas únicas detectadas: {len(detected_persons_set)}")
    
    if detected_persons_set:
        print()
        print("Personas detectadas:")
        for person_name in detected_persons_set:
            print(f"  • {person_name}")
    
    print("=" * 70)
    print()


def test_database():
    """Prueba las operaciones de la base de datos"""
    print("=" * 70)
    print("   TEST DE BASE DE DATOS")
    print("=" * 70)
    print()
    
    db_path = "data/faces.db"
    
    if not os.path.exists(db_path):
        print("⚠️  La base de datos no existe")
        print(f"   Ejecuta: python init_database.py")
        print()
        return
    
    # Conectar
    db = FaceDatabase(db_path)
    db.connect()
    
    # Obtener personas
    persons = db.get_all_persons()
    
    print(f"📊 Total de personas registradas: {len(persons)}")
    print()
    
    if persons:
        print("Lista de personas:")
        print("-" * 70)
        
        for person_id, name, img_path in persons:
            # Verificar si existe la imagen
            img_exists = "✅" if os.path.exists(img_path) else "❌"
            
            print(f"ID: {person_id:3d} | {img_exists} | {name}")
            print(f"           Imagen: {img_path}")
            print()
    else:
        print("⚠️  No hay personas registradas")
        print()
    
    # Cerrar
    db.close()
    
    print("=" * 70)
    print()


def main():
    """Menú principal"""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "TEST DE RECONOCIMIENTO FACIAL" + " " * 24 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("Opciones:")
    print("  1. Probar reconocimiento facial (10 segundos)")
    print("  2. Probar reconocimiento facial (30 segundos)")
    print("  3. Probar reconocimiento facial (tiempo personalizado)")
    print("  4. Ver base de datos")
    print("  5. Salir")
    print()
    
    while True:
        try:
            opcion = input("Selecciona una opción (1-5): ").strip()
            
            if opcion == "1":
                print()
                test_recognition(10)
                break
            elif opcion == "2":
                print()
                test_recognition(30)
                break
            elif opcion == "3":
                print()
                try:
                    segundos = int(input("¿Cuántos segundos? "))
                    if segundos > 0:
                        print()
                        test_recognition(segundos)
                        break
                    else:
                        print("⚠️  Debe ser un número positivo")
                except ValueError:
                    print("⚠️  Valor inválido")
            elif opcion == "4":
                print()
                test_database()
                break
            elif opcion == "5":
                print("\n👋 ¡Hasta luego!\n")
                break
            else:
                print("⚠️  Opción inválida")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Cancelado por el usuario\n")
            break


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
