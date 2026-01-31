"""
Script para inicializar la base de datos de reconocimiento facial
Permite registrar personas manualmente desde imágenes
"""

import cv2
import os
from face_database import FaceDatabase


def register_person_from_image(db, image_path, person_name):
    """
    Registra una persona desde una imagen
    
    Args:
        db: Instancia de FaceDatabase
        image_path: Ruta a la imagen con la cara
        person_name: Nombre de la persona
    """
    # Leer imagen
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"❌ Error: No se pudo leer la imagen {image_path}")
        return False
    
    # Registrar en la base de datos
    success = db.register_person(person_name, img, hash_size=16)
    
    if success:
        print(f"✅ {person_name} registrado exitosamente")
        return True
    else:
        print(f"⚠️  {person_name} ya está registrado")
        return False


def main():
    """Función principal"""
    print("=" * 60)
    print("   INICIALIZACIÓN DE BASE DE DATOS - RECONOCIMIENTO FACIAL")
    print("=" * 60)
    print()
    
    # Crear base de datos
    db_path = "data/faces.db"
    db = FaceDatabase(db_path, images_cache_dir="data/face_images")
    db.connect()
    
    print(f"✅ Base de datos creada en: {db_path}")
    print(f"✅ Caché de imágenes en: data/face_images")
    print()
    
    # Opción 1: Registrar desde directorio
    print("OPCIÓN 1: Registrar personas desde un directorio")
    print("-" * 60)
    
    faces_dir = "data/training_faces"
    
    if os.path.exists(faces_dir):
        print(f"Buscando imágenes en: {faces_dir}")
        print()
        
        # Buscar imágenes
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        
        for filename in os.listdir(faces_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                images.append(filename)
        
        if images:
            print(f"Se encontraron {len(images)} imágenes")
            print()
            
            for img_file in images:
                # Usar el nombre del archivo (sin extensión) como nombre de la persona
                person_name = os.path.splitext(img_file)[0].replace('_', ' ').title()
                img_path = os.path.join(faces_dir, img_file)
                
                register_person_from_image(db, img_path, person_name)
        else:
            print("⚠️  No se encontraron imágenes en el directorio")
    else:
        print(f"⚠️  El directorio {faces_dir} no existe")
        print(f"   Crea el directorio y coloca imágenes de caras allí")
        print(f"   Nombra las imágenes con el nombre de la persona (ejemplo: juan_perez.jpg)")
    
    print()
    print("-" * 60)
    
    # Opción 2: Registrar manualmente
    print()
    print("OPCIÓN 2: Registrar personas manualmente")
    print("-" * 60)
    
    while True:
        print()
        resp = input("¿Deseas registrar una persona manualmente? (s/n): ").strip().lower()
        
        if resp != 's':
            break
        
        person_name = input("Nombre de la persona: ").strip()
        
        if not person_name:
            print("❌ El nombre no puede estar vacío")
            continue
        
        img_path = input("Ruta a la imagen de la cara: ").strip()
        
        if not os.path.exists(img_path):
            print(f"❌ La imagen no existe: {img_path}")
            continue
        
        register_person_from_image(db, img_path, person_name)
    
    # Mostrar resumen
    print()
    print("=" * 60)
    print("   RESUMEN")
    print("=" * 60)
    
    persons = db.get_all_persons()
    
    if persons:
        print(f"\n✅ Total de personas registradas: {len(persons)}\n")
        
        for person_id, name, img_path in persons:
            print(f"  • {name}")
    else:
        print("\n⚠️  No hay personas registradas aún")
        print()
        print("INSTRUCCIONES:")
        print("1. Crea el directorio 'data/training_faces'")
        print("2. Coloca imágenes de caras en ese directorio")
        print("3. Nombra las imágenes con el nombre de la persona")
        print("4. Ejecuta este script nuevamente")
    
    print()
    
    # Cerrar base de datos
    db.close()
    
    print("✅ Base de datos cerrada")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Operación cancelada por el usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
