import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pydicom
import os
import nrrd

'''
This Python script automates the segmentation of medical images (DICOM files), 
specifically targeting high-intensity regions, likely bone or dense tissue. 
It applies different image processing techniques, including normalization, 
CLAHE enhancement, morphological operations, and Otsu's thresholding, based 
on the maximum pixel intensity of the input image. The final segmented masks 
are saved as both PNG and NRRD files.
'''

output_folder="otsu"
for i  in range(1,502):
    folder="dicom_data\\"+str(i)
    # Cargar archivo DICOM
    dicom_path = os.path.join(folder, 'dicomFile.dcm')

    if not os.path.isfile(dicom_path):
        print(f"ERROR: No existe el archivo DICOM: {dicom_path}")
        continue

    try:
        dicom_file = pydicom.dcmread(dicom_path)
        pixel_array = dicom_file.pixel_array
    except Exception as e:
        print(f"ERROR al leer DICOM {dicom_path}: {e}")
        continue
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

        # === APLICAR MÁSCARA A IMAGEN ORIGINAL ===
        result = cv.bitwise_and(img, img, mask=final_mask)
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
            # Encuentra las coordenadas del bounding box de la máscara
            x, y, w, h = cv.boundingRect(final_mask)

            # Recorta la imagen original (puede ser 'img', 'img_eq' o incluso 'pixel_array')
            cropped_image = img[y:y+h, x:x+w]

            # También puedes recortar la máscara si la necesitas
            cropped_mask = final_mask[y:y+h, x:x+w]

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



print("Segmentación completada. Imágenes guardadas en la carpeta 'otsu'.")