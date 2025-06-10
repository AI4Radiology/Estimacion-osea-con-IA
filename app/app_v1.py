# Version previa a evaluación con asesor Juan Orejuela y se tiene como evidencia de trabajo previo a correcciones
'''van Griethuysen, J. J. M., Fedorov, A., Parmar, C., Hosny, A., Aucoin, N., Narayan, V., Beets-Tan, R. G. H., Fillon-Robin, J. C., Pieper, S., Aerts, H. J. W. L. (2017). Computational Radiomics System to Decode the Radiographic Phenotype. Cancer Research, 77(21), e104–e107. https://doi.org/10.1158/0008-5472.CAN-17-0339 <https://doi.org/10.1158/0008-5472.CAN-17-0339>_'''

import os
import glob
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from segment_anything import SamAutomaticMaskGenerator
import torch
from segment_anything import sam_model_registry
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import radiomics
from radiomics import featureextractor
import csv
import SimpleITK as sitk
import numpy as np
import nrrd
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def load_images(folder_path, parent_window=None):
    # Pantalla de carga
    loading_win = None
    if parent_window:
        loading_win = tk.Toplevel(parent_window)
        loading_win.title("Cargando...")
        tk.Label(loading_win, text="Generando máscaras, por favor espera...").pack(padx=20, pady=20)
        loading_win.update()

    images = []    
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"
    CHECKPOINT_PATH = "sam_vit_h_4b8939.pth" 
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)

    dicom_file = pydicom.dcmread(os.path.join(folder_path, 'dicomFile.dcm'))
    image_array = dicom_file.pixel_array
    image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
    result = mask_generator.generate(image_bgr)

    output_folder = os.path.join(folder_path, 'segments')
    os.makedirs(output_folder, exist_ok=True)

    for i, mask in enumerate(result):
        mask_array = mask['segmentation'].astype(np.uint8)
        mask_image = cv2.cvtColor(mask_array * 255, cv2.COLOR_GRAY2BGR)
        output_path = os.path.join(output_folder, f'mask_{i}.png')
        cv2.imwrite(output_path, mask_image)

    segment_folder = os.path.join(folder_path, 'segments')
    for file in os.listdir(segment_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
            images.append(os.path.join(segment_folder, file))

    if loading_win:
        loading_win.destroy()
    return images

def mix_masks(folder_path):
    original_folder = folder_path
    
    folder_path = os.path.join(folder_path, 'segments')
    mask_paths = glob.glob(folder_path + "/**")
    try:
        final_mask = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)

        if final_mask is None:
            return None

        # Iterar sobre las demás máscaras y combinarlas
        for mask_path in mask_paths[1:]:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                continue

            final_mask = cv2.bitwise_or(final_mask, mask)

        # Guardar la máscara final en PNG
        output_png =  os.path.join(folder_path,"mascara_final.png")
        cv2.imwrite(output_png, final_mask)

        # Normalizar y guardar en formato NRRD
        if np.max(final_mask) > 0:  # Evita dividir por cero
            image_array = (final_mask > 128).astype(np.uint8)
            image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
            image_array = np.transpose(image_array)
            output_nrrd = os.path.join(original_folder, "segmentation.nrrd")
            nrrd.write(output_nrrd, image_array)
            print(f"Éxito: Máscara combinada guardada en {output_nrrd}")
        else:
            print(f"Advertencia: Máscara vacía en {folder_path}, no se generó NRRD.")

    except Exception as e:
        print(f"Error en {folder_path}: {e}")


def mix_images(folder_path, selected_images):
    folder_path = os.path.join(folder_path, 'segments')
    mask_paths = glob.glob(folder_path + "/*.png")
    if not mask_paths:
        return None
    
    first_mask = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)
    image_shape = first_mask.shape if first_mask is not None else (300, 300)
    image = np.zeros(image_shape, dtype=np.uint8) 
    for mask_path in mask_paths[0:]:
        if mask_path not in selected_images:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.bitwise_or(image, mask)
    
    return image




def extract_masks(folder_path, parent_window=None):
    # Pantalla de carga
    loading_win = None
    if parent_window:
        loading_win = tk.Toplevel(parent_window)
        loading_win.title("Procesando...")
        tk.Label(loading_win, text="Extrayendo características radiómicas...").pack(padx=20, pady=20)
        loading_win.update()

    mix_masks(folder_path)
    params_file = os.path.abspath(os.path.join('params', 'Params.yaml'))
    extractor = featureextractor.RadiomicsFeatureExtractor(params_file)

    dicom_file = os.path.join(folder_path, "dicomFile.dcm")
    image_path = os.path.join(folder_path, 'nrrdFile.nrrd')
    image = sitk.ReadImage(dicom_file)
    sitk.WriteImage(image, image_path)

    mask_path = os.path.join(folder_path, 'segmentation.nrrd')
    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        print(f"Advertencia: Archivos no encontrados en la carpeta {folder_path}")
        if loading_win:
            loading_win.destroy()
        return None

    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)
    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)

    mask_3d = np.stack([mask_array] * image_array.shape[0], axis=0)
    mask_3d_sitk = sitk.GetImageFromArray(mask_3d)
    mask_3d_sitk.CopyInformation(image)

    new_mask_path = mask_path.replace(".nrrd", "_3D.nrrd")
    sitk.WriteImage(mask_3d_sitk, new_mask_path)

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

    scaler = joblib.load("scaler.pkl")

    x = scaler.transform(x)
    print(len(x[0]))
    lasso_cv = joblib.load('lasso_model.pkl')
    
    predicciones = lasso_cv.predict(x)

    return predicciones

def show_images():
    folder_path = filedialog.askdirectory(title="Selecciona una carpeta")
    if not folder_path:
        return
    
    images = load_images(folder_path, parent_window=main_root)
    if not images:
        messagebox.showinfo("Información", "No se encontraron imágenes en la carpeta seleccionada.")
        return
    
    def toggle_selection(index):
        if index in selected_images:
            selected_images.remove(index)
            buttons[index].config(relief=tk.RAISED)
        else:
            selected_images.add(index)
            buttons[index].config(relief=tk.SUNKEN)
        show_mixed_image()
    
    def mostrar_edad_estimacion(folder_path, ageI, gender):
        age = predict_age(folder_path, ageI, gender)
        delta = timedelta(days=age[0])
        start_date = datetime(1, 1, 1)
        end_date = start_date + delta
        diff = relativedelta(end_date, start_date)
        messagebox.showinfo("Edad Estimada",
                            f"Edad estimada:\n{diff.years} años, {diff.months} meses, {diff.days} días")
    
    def show_mixed_image():
        mixed_img = mix_images(folder_path, {images[i] for i in selected_images})
        if mixed_img is not None:
            mixed_img = cv2.cvtColor(mixed_img, cv2.COLOR_GRAY2RGB)
            mixed_img = Image.fromarray(mixed_img)
            mixed_img.thumbnail((300, 300))
            img_tk = ImageTk.PhotoImage(mixed_img)
            selected_label.config(image=img_tk)
            selected_label.image = img_tk
    
    def delete_selected():
        if not selected_images:
            messagebox.showwarning("Advertencia", "No has seleccionado ninguna imagen para eliminar.")
            return
        
        confirm = messagebox.askyesno("Confirmar", "¿Seguro que quieres eliminar las imágenes seleccionadas?")
        if confirm:
            for index in selected_images:
                os.remove(images[index])
            messagebox.showinfo("Éxito", "Imágenes eliminadas correctamente.")
            extract_masks(folder_path, parent_window=main_root)
            age=predict_age(folder_path,5082,1)

            delta = timedelta(days=age[0])

            start_date = datetime(1, 1, 1)

            # Fecha final sumando los días
            end_date = start_date + delta

            # Calcular la diferencia usando relativedelta
            diff = relativedelta(end_date, start_date)
            
            print(f"Años: {diff.years}, Meses: {diff.months}, Días: {diff.days}")

            mostrar_edad_estimacion(folder_path,5082,1)

            root.destroy()
            main_root.destroy()
    
    root = tk.Toplevel()
    root.title("Seleccionar imágenes a eliminar")
    root.geometry("890x400")
    root.protocol("WM_DELETE_WINDOW", lambda: [root.destroy(), main_root.destroy()])
    
    frame = tk.Frame(root)
    frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    canvas = tk.Canvas(frame)
    scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas)
    
    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )
    
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    selected_images = set()
    buttons = []
    
    for idx, img_path in enumerate(images):
        img = Image.open(img_path)
        img.thumbnail((100, 100))
        img_tk = ImageTk.PhotoImage(img)
        
        btn = tk.Button(scroll_frame, image=img_tk, command=lambda i=idx: toggle_selection(i))
        btn.image = img_tk
        btn.grid(row=idx // 5, column=idx % 5, padx=5, pady=5)
        
        buttons.append(btn)
    
    
    del_button = tk.Button(root, text="Eliminar seleccionadas", command=delete_selected, bg="red", fg="white")
    del_button.pack(side=tk.BOTTOM, pady=10)
    
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    mixed_img = mix_images(folder_path, {})

    mixed_img = cv2.cvtColor(mixed_img, cv2.COLOR_GRAY2RGB)
    mixed_img = Image.fromarray(mixed_img)
    mixed_img.thumbnail((300, 300))
    img_tk = ImageTk.PhotoImage(mixed_img)
    
    selected_label = tk.Label(root, image=img_tk)
    
    selected_label.image = img_tk
    selected_label.pack(side=tk.RIGHT, padx=10, pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    main_root = tk.Tk()
    main_root.withdraw()
    show_images()
    
