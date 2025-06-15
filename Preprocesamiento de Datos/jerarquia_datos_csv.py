import os
import csv
import xml.etree.ElementTree as ET
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

#@author Eric Cabrera Cruz

def process_from_csv(csv_path):

    """
    Procesa un archivo CSV con metadatos asociados a una imagen MRI, extrae los más importantes y llama a la función para procesar cada imagen.

    Args:
        csv_path (str): Ruta del archivo CSV con los metadatos.

    Returns:
        None
    """

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        subject_identifier = row["Subject"]
        visit_identifier = row["Visit"]
        image_uid = row["Image Data ID"]
        original_date = row["Acq Date"]
        fecha_dt = datetime.strptime(original_date, "%m/%d/%Y")
        date_acquired = fecha_dt.strftime("%Y-%m-%d")
        data_label = row["Description"]
        research_group = row["Group"]

        print(f"\nProcesando: {subject_identifier}, {image_uid}")
        process_neuro_images(subject_identifier, visit_identifier, image_uid, date_acquired, data_label, research_group)


def load_neuro_image(path):

    """
    Carga una imagen MRI en formato .nii.

    Args:
        path (str): Ruta al archivo de imagen .nii.

    Returns:
        nibabel.Nifti1Image or None: Objeto imagen cargado si existe y se puede abrir, sino None.
    """

    if not os.path.exists(path):
        print(f"Error: No existe el archivo {path}")
        return None
    try:
        return nib.load(path)
    except Exception as e:
        print(f"Error al cargar la imagen {path}: {e}")
        return None

def show_and_save_image(neuro_image_data, output_dir, subject_identifier, image_uid, visit_identifier, category, plane, index):
    
    """
        Muestra un corte específico de la imagen MRI y la guarda como archivo JPG.

        Args:
            neuro_image_data (numpy.ndarray): Matriz 3D con los datos de la imagen.
            output_dir (str): Directorio base para guardar las imágenes.
            subject_identifier (str): Identificador del sujeto.
            image_uid (str): Identificador único de la imagen.
            visit_identifier (str): Identificador de la visita.
            category (str): Categoría del sujeto (MCI, CN, AD).
            plane (str): Plano del corte ('Axial', 'Coronal' o 'Sagittal').
            index (int): Índice del corte en el plano seleccionado.

        Raises:
            ValueError: Si el plano no es válido.

        Returns:
            str: Ruta del archivo JPG guardado.
    """

    visit_dir = os.path.join(output_dir, visit_identifier)
    os.makedirs(visit_dir, exist_ok=True)

    plane_dir = os.path.join(visit_dir, plane)
    os.makedirs(plane_dir, exist_ok=True)
    category_dir = os.path.join(plane_dir, category)
    os.makedirs(category_dir, exist_ok=True)
    
    
    
    if plane == 'Axial':
        data_slice = neuro_image_data[index, :, :]
    elif plane == 'Coronal':
        data_slice = neuro_image_data[:, index, :]
    elif plane == 'Sagittal':
        data_slice = neuro_image_data[:, :, index]
    else:
        raise ValueError("Plano no válido. Elija entre 'sagittal', 'coronal' o 'axial'.")

    file_name = f"{subject_identifier}_{image_uid}_{plane}_{index}.jpg"
    file_path = os.path.join(category_dir, file_name)

    plt.imshow(data_slice, cmap='bone')
    plt.axis('off')

    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return file_path


def process_neuro_images(subject_identifier, visit_identifier, image_uid, date_acquired, processed_data_label, research_group):

    """
    Busca y procesa las imágenes MRI de un sujeto en función de sus metadatos y guarda sus cortes en JPG.

    Args:
        subject_identifier (str): Identificador del sujeto.
        visit_identifier (str): Identificador de la visita.
        image_uid (str): Identificador único de la imagen.
        date_acquired (str): Fecha de adquisición en formato 'YYYY-MM-DD'.
        processed_data_label (str): Etiqueta que describe el procesamiento de la imagen.
        research_group (str): Grupo de investigación o categoría del sujeto (MCI, CN, AD).

    Returns:
        None
    """


    base_path = os.path.abspath( r"D:\data\Final Dataset\ADNI")
    subject_folder = os.path.join(base_path, subject_identifier)

    if processed_data_label == "MPR; GradWarp; B1 Correction; N3; Scaled":
        processed_folder = "MPR__GradWarp__B1_Correction__N3__Scaled"
    elif processed_data_label == "MPR; GradWarp; B1 Correction":
        processed_folder = "MPR__GradWarp__B1_Correction"
    elif processed_data_label == "MPR; GradWarp; N3; Scaled":
        processed_folder = "MPR__GradWarp__N3__Scaled"
    elif processed_data_label == "MPR; GradWarp; N3":
        processed_folder = "MPR__GradWarp__N3"
    elif processed_data_label == "MPR; ; N3; Scaled":
        processed_folder = "MPR____N3__Scaled"
    elif processed_data_label == "MPR; ; N3; Scaled_2":
        processed_folder = "MPR____N3__Scaled_2"
    elif processed_data_label == "MPR; GradWarp; B1 Correction; N3; Scaled_2":
        processed_folder = "MPR__GradWarp__B1_Correction__N3__Scaled_2"
    elif processed_data_label == "MPR; GradWarp; B1 Correction; N3":
        processed_folder = "MPR__GradWarp__B1_Correction__N3"
    elif processed_data_label == "MPR-R; GradWarp; B1 Correction; N3; Scaled":
        processed_folder = "MPR-R__GradWarp__B1_Correction__N3__Scaled"
    else:
        print(f"Error: processed_data_label '{processed_data_label}' no válido.")
        return

    date_folder = None
    for folder in os.listdir(os.path.join(subject_folder, processed_folder)):
        if folder.startswith(date_acquired):
            potential_path = os.path.join(subject_folder, processed_folder, folder, image_uid)
            if os.path.isdir(potential_path):
                date_folder = folder
                break
    if not date_folder:
        print(f"Error: No se encontró la carpeta de la fecha {date_acquired} en {subject_folder}/{processed_folder}")
        return
    
    image_folder = os.path.join(subject_folder, processed_folder, date_folder, f"{image_uid}")
    nii_files = [f for f in os.listdir(image_folder) if f.endswith(".nii")]
    
    if not nii_files:
        print(f"Error: No se encontró el archivo .nii en {image_folder}")
        return
    
    nii_path = os.path.join(image_folder, nii_files[0])
    
    neuro_image = load_neuro_image(nii_path)
    if not neuro_image:
        return

    neuro_image_data = neuro_image.get_fdata()

    category_map = {"MCI": "MCI", "CN": "CN", "AD": "AD"}
    category = category_map.get(research_group, "Otros")
    
    dataset_dir = os.path.abspath(r"D:\data\Datase_ordernado_visitas")
    os.makedirs(dataset_dir, exist_ok=True)

    views = ['Sagittal', 'Coronal', 'Axial']
    for view in views:
        for index in range(neuro_image_data.shape[2]):
            max_index = neuro_image_data.shape[2] - neuro_image_data.shape[2]*0.25
            min_index = neuro_image_data.shape[2]*0.25
            
            if index > min_index and index < max_index:
                image_path = show_and_save_image(neuro_image_data, dataset_dir, subject_identifier, image_uid, visit_identifier, category, view, index)
                print(f"Guardada imagen: {image_path}")
            else:
                continue
    print(f"Imágenes de {subject_identifier} guardadas en {dataset_dir}")

base_path = r"D:\data\Final Dataset\ADNI"
csv_input = r"C:\Users\Eric\Desktop\Novus Initium\Ars Discendi\TFG\Metadatos_finales.csv"

process_from_csv(csv_input)
