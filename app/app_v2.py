import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from tkcalendar import DateEntry
from datetime import datetime
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pydicom
import gdcm
pydicom.config.use_gdcm = False

import os
import nrrd
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import sys
from PIL import Image, ImageTk
import csv
import SimpleITK as sitk
'''van Griethuysen, J. J. M., Fedorov, A., Parmar, C., Hosny, A., Aucoin, N., Narayan, V., Beets-Tan, R. G. H., Fillon-Robin, J. C., Pieper, S., Aerts, H. J. W. L. (2017). Computational Radiomics System to Decode the Radiographic Phenotype. Cancer Research, 77(21), e104–e107. https://doi.org/10.1158/0008-5472.CAN-17-0339 <https://doi.org/10.1158/0008-5472.CAN-17-0339>_'''
from radiomics import featureextractor

if getattr(sys, 'frozen', False):  # solo si está empaquetado
    base_path = sys._MEIPASS
    dll_path = os.path.join(base_path, "xgboost", "lib")
    os.environ["PATH"] = dll_path + os.pathsep + os.environ["PATH"]
    
def obtener_ruta_recurso(ruta_relativa):
    """Obtiene la ruta de un recurso, compatible con PyInstaller (modo --onefile)."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, ruta_relativa)
    return os.path.join(os.path.abspath("."), ruta_relativa)

def extract_masks(folder_path, parent_window=None):
    loading_win = None
    if parent_window:
        loading_win = tk.Toplevel(parent_window)
        loading_win.title("Procesando...")
        tk.Label(loading_win, text="Extrayendo características radiómicas...").pack(padx=20, pady=20)
        loading_win.update()

    # Parámetros y rutas
    params_file = obtener_ruta_recurso(os.path.join('params', 'Params.yaml'))
    extractor = featureextractor.RadiomicsFeatureExtractor(params_file)

    dicom_file = os.path.join(folder_path, "dicomFile.dcm")
    image_path = os.path.join(folder_path, 'nrrdFile.nrrd')
    mask_path = os.path.join(folder_path, 'segmentation.nrrd')

    # Guardar imagen como .nrrd
    image = sitk.ReadImage(dicom_file)
    sitk.WriteImage(image, image_path)

    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        print(f"Advertencia: Archivos no encontrados en {folder_path}")
        if loading_win:
            loading_win.destroy()
        return None

    # Cargar imagen y máscara
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)
    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)

    # Crear máscara 3D
    mask_3d = np.stack([mask_array] * image_array.shape[0], axis=0)
    mask_3d_sitk = sitk.GetImageFromArray(mask_3d)
    mask_3d_sitk.CopyInformation(image)

    new_mask_path = mask_path.replace(".nrrd", "_3D.nrrd")
    sitk.WriteImage(mask_3d_sitk, new_mask_path)

    # Extraer características
    output_csv = os.path.join(folder_path, 'radiomics_features.csv')
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=';')
        feature_vector = extractor.execute(image_path, new_mask_path)
        header = list(feature_vector.keys())
        row = list(feature_vector.values())
        csv_writer.writerow(header)
        csv_writer.writerow(row)

    if loading_win:
        loading_win.destroy()
    return output_csv

def mostrar_imagen(imagen_path):
    ventana_img = tk.Toplevel(ventana)
    ventana_img.title("Imagen Segmentada")
    
    img = Image.open(imagen_path)
    img = img.resize((350, 350))  # Redimensiona si es necesario
    photo = ImageTk.PhotoImage(img)

    lbl = tk.Label(ventana_img, image=photo)
    lbl.image = photo  # mantener referencia
    lbl.pack(padx=10, pady=10)


# Métodos simulados
def segment(file_path):
    folder = os.path.dirname(file_path)
    # Cargar archivo DICOM
    dicom_path = file_path

    if not os.path.isfile(dicom_path):
        print(f"ERROR: No existe el archivo DICOM: {dicom_path}")

    try:
        dicom_file = pydicom.dcmread(dicom_path)
        pixel_array = dicom_file.pixel_array
    except Exception as e:
        print(f"ERROR al leer DICOM {dicom_path}: {e}")

    max_val = pixel_array.max()
    # Normalización por percentil
    p_low, p_high = np.percentile(pixel_array, [2, 98])
    clipped = np.clip(pixel_array, p_low, p_high)
    img = ((clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)

    if max_val <= 5000:
        clahe = cv.createCLAHE(clipLimit=1.5, tileGridSize=(4,4))
        img_eq = clahe.apply(img)
        kernel_tophat = cv.getStructuringElement(cv.MORPH_RECT, (180, 180))
        tophat = cv.morphologyEx(img_eq, cv.MORPH_TOPHAT, kernel_tophat)
        _, mask = cv.threshold(tophat, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # === MORFOLOGÍA PARA LIMPIEZA ===
        kernel_morph = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        mask_clean = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_morph, iterations=1)
        mask_clean = cv.morphologyEx(mask_clean, cv.MORPH_CLOSE, kernel_morph, iterations=1)

        # === FILTRAR CONTORNOS PEQUEÑOS ===
        contours, _ = cv.findContours(mask_clean, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(mask)
        for cnt in contours:
            if cv.contourArea(cnt) > 500:
                cv.drawContours(final_mask, [cnt], -1, 255, -1)

    else:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_eq = clahe.apply(img)
        blur = cv.GaussianBlur(img_eq, (31, 31), 0)
        _, mask = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # Morfología + contornos grandes
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 1))
        mask_clean = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)
        mask_clean = cv.morphologyEx(mask_clean, cv.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv.findContours(mask_clean, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(mask)

        for cnt in contours:
            if cv.contourArea(cnt) > 500:
                cv.drawContours(final_mask, [cnt], -1, 255, -1)
        max_val = pixel_array.max()
        # Normalización por percentil
        p_low, p_high = np.percentile(pixel_array, [1, 99])
        clipped = np.clip(pixel_array, p_low, p_high)
        img = ((clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)

        if np.any(final_mask):

            segmented_image = cv.bitwise_and(img, img, mask=final_mask)
            masked_values = segmented_image[final_mask == 255]

            # Calcula el umbral de Otsu solo con esos valores
            otsu_thresh = cv.threshold(masked_values, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[0]

            # Aplica el umbral a toda la imagen segmentada
            otsu_mask = np.zeros_like(segmented_image)
            otsu_mask[segmented_image > otsu_thresh] = 255

            _, otsu_mask2 = cv.threshold(segmented_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            segmented_image = cv.bitwise_and(img, img, mask=otsu_mask2)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
            bone_mask_cleaned = cv.morphologyEx(segmented_image, cv.MORPH_OPEN, kernel, iterations=2)
            _, otsu_mask2 = cv.threshold(bone_mask_cleaned, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            segmented_image = cv.bitwise_and(img, img, mask=otsu_mask2)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
            bone_mask_cleaned = cv.morphologyEx(segmented_image, cv.MORPH_OPEN, kernel, iterations=2)
            _, otsu_mask2 = cv.threshold(bone_mask_cleaned, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            segmented_image = cv.bitwise_and(img, img, mask=otsu_mask2)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
            bone_mask_cleaned = cv.morphologyEx(segmented_image, cv.MORPH_OPEN, kernel, iterations=2)
            _, otsu_mask2 = cv.threshold(bone_mask_cleaned, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            segmented_image = cv.bitwise_and(img, img, mask=otsu_mask2)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
            bone_mask_cleaned = cv.morphologyEx(segmented_image, cv.MORPH_OPEN, kernel, iterations=2)
            _, otsu_mask2 = cv.threshold(bone_mask_cleaned, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

            _, final_mask = cv.threshold(segmented_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            blurred_segmented = cv.GaussianBlur(segmented_image, (31, 31), 0)

            # Luego continúa con el mismo procedimiento:
            masked_values = blurred_segmented[final_mask == 255]
            otsu_thresh = cv.threshold(masked_values, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[0]
            otsu_mask = np.zeros_like(segmented_image)
            otsu_mask[blurred_segmented > otsu_thresh] = 255
    

    # === PASO 5: GUARDAR RESULTADOS ===
    cv.imwrite(os.path.join(folder, f'3_mask_alta_intensidad.png'), final_mask)

    if np.max(final_mask) > 0:  # Evita dividir por cero
            image_array = (final_mask > 128).astype(np.uint8)
            image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
            image_array = np.transpose(image_array)
            output_nrrd = os.path.join(folder, "segmentation.nrrd")
            nrrd.write(output_nrrd, image_array)
            print(f"Éxito: Máscara combinada guardada en {output_nrrd}")

def predict_age(folder_path,age,gender):
    import joblib
    # Cargar el modelo desde el archivo
    fileOrigin =os.path.join(folder_path, "radiomics_features.csv")

    df=pd.read_csv(fileOrigin, sep=';')

    campos_a_eliminar = [
    "diagnostics_Versions_Numpy",
    "diagnostics_Versions_SimpleITK",
    "diagnostics_Versions_PyWavelet",
    "diagnostics_Versions_Python",
    "diagnostics_Configuration_Settings",
    "diagnostics_Configuration_EnabledImageTypes",
    "diagnostics_Image-original_Hash",
    "diagnostics_Image-original_Dimensionality",
    "diagnostics_Image-original_Spacing",
    "diagnostics_Image-original_Size",
    "diagnostics_Image-original_Mean",
    "diagnostics_Image-original_Minimum",
    "diagnostics_Image-original_Maximum",
    "diagnostics_Mask-original_Hash",
    "diagnostics_Mask-original_Spacing",
    "diagnostics_Mask-original_Size",
    "diagnostics_Mask-original_BoundingBox",
    "diagnostics_Mask-original_VoxelNum",
    "diagnostics_Mask-original_VolumeNum",
    "diagnostics_Mask-original_CenterOfMassIndex",
    "diagnostics_Mask-original_CenterOfMass",
    "original_shape_Flatness", 
    "original_shape_LeastAxisLength", 
    "diagnostics_Versions_PyRadiomics"
    ]
    df.drop(columns=campos_a_eliminar, inplace=True)
    
    x=df.copy().values
    data=[gender, age]
    x = np.append(data,x)
    x=[x]

    scaler = joblib.load(obtener_ruta_recurso("scaler.pkl"))

    x = scaler.transform(x)
    print(len(x[0]))
    lasso_cv = joblib.load(obtener_ruta_recurso("xgboost_g.pkl"))
    
    predicciones = lasso_cv.predict(x)

    return predicciones

# Cálculo de edad
def calcular_edad(fecha_nacimiento, fecha_estudio):
    diferencia = fecha_estudio - fecha_nacimiento
    return diferencia.days

def procesar_datos():
    fecha_nacimiento = nacimiento_entry.get_date()
    fecha_estudio = estudio_entry.get_date()
    genero_texto = genero_var.get()
    archivo = entry_archivo.get()

    if not (fecha_nacimiento and fecha_estudio and genero_texto and archivo):
        messagebox.showerror("Error", "Todos los campos deben ser completados.")
        return

    if genero_texto == "Masculino":
        genero = 0
    elif genero_texto == "Femenino":
        genero = 1
    else:
        genero = -1 

    edad = calcular_edad(fecha_nacimiento, fecha_estudio)

    try:
        folder = os.path.dirname(archivo)

        # Paso 1: Segmentar
        segment(archivo)

        # Paso 2: Mostrar imagen segmentada inmediatamente
        mask_path = os.path.join(folder, '3_mask_alta_intensidad.png')
        if os.path.exists(mask_path):
            mostrar_imagen(mask_path)
            ventana.update()  # Actualiza la GUI para que la imagen se muestre antes de continuar

        # Paso 3: Extraer características con ventana de carga
        extract_masks(folder, parent_window=ventana)

        # Paso 4: Predecir edad
        resultado = predict_age(folder, edad, genero)

        delta = timedelta(days=float(resultado[0]))
        start_date = datetime(1, 1, 1)
        end_date = start_date + delta
        diff = relativedelta(end_date, start_date)

        resultado_text.set(
            f"Edad estimada:\n{diff.years} años, {diff.months} meses, {diff.days} días"
        )
    except Exception as e:
        messagebox.showerror("Error en procesamiento", str(e))

def seleccionar_archivo():
    path = filedialog.askopenfilename(title="Selecciona el archivo a segmentar")
    if path:
        entry_archivo.delete(0, tk.END)
        entry_archivo.insert(0, path)

# Crear ventana principal
ventana = tk.Tk()
ventana.title("Segmentación y Predicción de Edad")
ventana.geometry("400x500")
ventana.resizable(False, False)

# Fecha de nacimiento
tk.Label(ventana, text="Fecha de nacimiento:").pack(pady=(10,0))
nacimiento_entry = DateEntry(ventana, date_pattern='yyyy-mm-dd')
nacimiento_entry.pack()

# Fecha de estudio
tk.Label(ventana, text="Fecha de estudio:").pack(pady=(10,0))
estudio_entry = DateEntry(ventana, date_pattern='yyyy-mm-dd')
estudio_entry.pack()

# Género
tk.Label(ventana, text="Género:").pack(pady=(10,0))
genero_var = tk.StringVar()
combo_genero = ttk.Combobox(ventana, textvariable=genero_var, state="readonly")
combo_genero['values'] = ("Masculino", "Femenino")
combo_genero.pack()

# Archivo
tk.Label(ventana, text="Archivo a segmentar:").pack(pady=(10,0))
entry_archivo = tk.Entry(ventana, width=40)
entry_archivo.pack()
tk.Button(ventana, text="Buscar...", command=seleccionar_archivo).pack(pady=5)

# Botón de procesar
tk.Button(ventana, text="Procesar", command=procesar_datos).pack(pady=15)

# Resultado
resultado_text = tk.StringVar()
resultado_label = tk.Label(ventana, textvariable=resultado_text, justify="left", wraplength=350, fg="blue")
resultado_label.pack(pady=10)

ventana.mainloop()
