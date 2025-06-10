import os
import shutil

# Directorio raíz original y destino
origen = r"D:\data\los q faltan\ADNI"
destino = r"D:\data\directorio_filtrado\ADNI"

# Lista de descripciones válidas, usando formato de nombre de directorio (con '__')
descripciones_validas = {
    "MPR__GradWarp__B1_Correction__N3__Scaled",
    "MPR__GradWarp__N3__Scaled",
    "MPR__GradWarp__N3",
    "MPR____N3__Scaled",
    "MPR____N3__Scaled_2",
    "MPR__GradWarp__B1_Correction",
    "MPR__GradWarp__B1_Correction__N3__Scaled_2",
    "MPR__GradWarp__B1_Correction__N3",
    "MPR-R__GradWarp__B1_Correction__N3__Scaled",
    "MPR-R__GradWarp__B1_Correction",

}

# Recorremos los sujetos
for sujeto in os.listdir(origen):
    path_sujeto = os.path.join(origen, sujeto)
    if not os.path.isdir(path_sujeto):
        continue

    # Recorremos las carpetas de descripción dentro del sujeto
    for descripcion in os.listdir(path_sujeto):
        path_descripcion = os.path.join(path_sujeto, descripcion)
        if not os.path.isdir(path_descripcion):
            continue

        if descripcion in descripciones_validas:
            # Construir el path de destino
            destino_path = os.path.join(destino, sujeto, descripcion)
            if not os.path.exists(destino_path):
                shutil.copytree(path_descripcion, destino_path)
                print(f"Copiado: {path_descripcion} -> {destino_path}")

