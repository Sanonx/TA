import os
from pathlib import Path
import shutil

def find_first_parent_dir_of_artifacts(top_dir):
    for path in Path(top_dir).rglob('artifacts'):
        if path.is_dir():
            return path.parent.parent.parent  # Tomamos el padre tres veces
    return None

def copy_all_subdirs_to_new_location(source_dir, destination_dir):
    for item in source_dir.iterdir():
        if item.is_dir():
            # Comprueba si el directorio ya existe en el destino
            if not (destination_dir / item.name).exists():
                shutil.copytree(item, destination_dir / item.name)
                print(f"La carpeta {item} ha sido copiada a {destination_dir / item.name}")
            else:
                print(f"La carpeta {item} ya existe en {destination_dir / item.name}. No se copió.")

def main():
    top_dir = "\"https:"  # Reemplaza esto con el nombre de la carpeta superior desde la que quieres empezar a buscar

    first_artifacts_parent_dir = find_first_parent_dir_of_artifacts(top_dir)

    if first_artifacts_parent_dir:
        print(f"El directorio superior al que contiene 'artifacts' es: {first_artifacts_parent_dir}")

        directories = [item for item in first_artifacts_parent_dir.iterdir() if item.is_dir()]

        destination_dir = Path('runsMLFLOW')
        destination_dir.mkdir(parents=True, exist_ok=True)  # Crear el directorio runsMLFLOW si no existe

        for dir in directories:
            copy_all_subdirs_to_new_location(dir, destination_dir)

    else:
        print("No encontré ninguna carpeta que contenga una subcarpeta 'artifacts'.")

if __name__ == "__main__":
    main()

