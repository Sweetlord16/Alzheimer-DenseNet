import os

dir1 = r"D:\data\directorio_filtrado\ADNI"
dir2 = r"D:\data\los q faltan\ADNI"

subjects1 = set(os.listdir(dir1))
subjects2 = set(os.listdir(dir2))

subjects1 = {s for s in subjects1 if os.path.isdir(os.path.join(dir1, s))}
subjects2 = {s for s in subjects2 if os.path.isdir(os.path.join(dir2, s))}

solo_en_dir1 = subjects1 - subjects2
solo_en_dir2 = subjects2 - subjects1

faltantes = sorted(solo_en_dir1.union(solo_en_dir2))
print(len(faltantes))
# Guardar en un archivo .txt separado por comas
with open("D:\data\pacientes_faltantes.txt", "w") as f:
    f.write(",".join(faltantes))

print("Faltantes guardados en 'pacientes_faltantes.txt'")

