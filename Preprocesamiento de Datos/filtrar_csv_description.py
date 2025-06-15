import pandas as pd

#@author Eric Cabrera Cruz
# Nombre del archivo original de metadatos ADNI con todos los formatos
input_file = r"C:\Users\Eric\Desktop\Novus Initium\Ars Discendi\TFG\Csvs Metadata\Sin filtro\m06,sc,m12_4_29_2025.csv"

df = pd.read_csv(input_file)

# Lista de valores permitidos en la columna 'Description' asociados al formato NiFTI
valores_validos = [
    "MPR; GradWarp; B1 Correction; N3; Scaled",
    "MPR; GradWarp; B1 Correction",
    "MPR; GradWarp; N3; Scaled"
    "MPR; GradWarp; N3",
    "MPR; ; N3; Scaled",
    "MPR; ; N3; Scaled_2",
    "MPR; GradWarp; B1 Correction; N3; Scaled_2",
    "MPR; GradWarp; B1 Correction; N3",
    "MPR-R; GradWarp; B1 Correction; N3; Scaled",
]

filtered_df = df[df['Description'].isin(valores_validos)]




# Guardamos el resultado en un nuevo CSV
output_file = r"C:\Users\Eric\Desktop\Novus Initium\Ars Discendi\TFG\Csvs Metadata\Con filtro\subjects_left_m06_sc_m12_filtered_output.csv"
filtered_df.to_csv(output_file, index=False)

print(f"Archivo guardado como '{output_file}' con {len(filtered_df)} filas filtradas.")
