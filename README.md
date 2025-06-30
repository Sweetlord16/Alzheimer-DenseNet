#  Manual de Usuario

Este repositorio contiene el código necesario para llevar a cabo simulaciones de trabajo realizado en este TFG, utilizando la arquitectura de redes convolucionales 2D DenseNet. El proceso incluye tanto el **preprocesamiento del conjunto de datos para la entrada del sistema** como la **ejecución de los distintos modelos de clasificación**.

---

##  Estructura del Proyecto

El proyecto se divide en dos bloques principales:

---

## 1. Preprocesamiento del Conjunto de Datos

###  Objetivo
Transformar los datos brutos descargados desde la plataforma **IDA de ADNI** en un conjunto listo para su uso por los modelos de clasificación.

###  Requisitos de Entrada
- Archivos `.nii` en formato **NIFTI**
- Archivos `.csv` con los metadatos de los .nii
- Archivo `ROOSTER.CSV` para mapear RID ↔ PTID

###  Flujo de Ejecución

1. **Mapeo RID a PTID**  
   Convierte los RIDs en PTIDs usando el archivo `ROOSTER.CSV`.

   ```bash
   python mapeo_rid_ptid.py
   ```

2. **Filtrado y procesamiento**
   - Filtrado de metadatos:
     ```bash
     python filtrar_csv_description.py
     python filtro_metadatos_nii.py
     ```
   - Filtrado de datos faltantes:
     ```bash
     python sujetos_faltantes_adni.py
     ```
   - Reorganización final del conjunto de datos:
     ```bash
     python jerarquia_datos_csv.py
     ```

###  Resultado Esperado
- Carpeta con imágenes `.jpg` generadas a partir de los `.nii` ordenadas por planos de visualización y marcas temporales
- CSV con metadatos filtrados y estructurados

---

##  2. Ejecución de Modelos de Clasificación

###  Arquitecturas Disponibles

| Archivo                          | Arquitectura        | Preentrenado | Fuente de datos         | K-Fold |
|----------------------------------|---------------------|--------------|--------------------------|--------|
| `DenseNet Final.py`              | DenseNet121         |  Sí          | `screening/m12`          | 5      |
| `Densenet No preentrenada.py`    | DenseNet custom     |  No          | `screening/m12`          | 2      |
| `Diagnosis_sc_m06.py`            | DenseNet121         |  Sí          | `diagnosis/m06 + sc`     | 5      |
| `Diagnosis_sc_m06_m12_k5.py`     | DenseNet121         |  Sí          | `diagnosis/m06 + sc`     | 5      |

###  Rutas de Datos

#### Screening (m12)
```python
trainin_paths = [
    '.../m12/Axial',
    '.../m12/Coronal',
    '.../m12/Sagittal'
]
```

#### Diagnosis (m06 + sc)
```python
trainin_paths = [
    '.../m06/Axial',
    '.../m06/Coronal',
    '.../m06/Sagittal'
]

diagnosis_training_path = [
    '.../sc/Axial',
    '.../sc/Coronal',
    '.../sc/Sagittal'
]
```

###  Ejecución

Ejemplo de ejecución de modelo:
```bash
python "DenseNet Final.py"
```

Cada código realiza entrenamiento por planos (Axial, Coronal, Sagittal) con validación cruzada.  

---

##  Notas Adicionales

- Todos los modelos utilizan `get_image_paths_and_labels()` para cargar datos.
- El usuario debe descargar manualmente los `.nii` desde la plataforma de ADNI.
- Las imágenes se transforman internamente a `.jpg` para su uso en entrenamiento.

---

##  Instalación y Dependencias

### Requisitos
- Python >= 3.8
- Librerías:
  - `torch`
  - `scikit-learn`
  - `nibabel`
  - `opencv-python`
  - `pandas`
  - `numpy`

### Instalación

Se recomienda usar un entorno virtual, en nuestro caso se hizo uso del entorno CONDA. 

##  Resultados

Durante la ejecución se imprimen métricas por fold. Las gráficas pueden descargarse en el drive adjunto

---
