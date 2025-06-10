import os 
'''van Griethuysen, J. J. M., Fedorov, A., Parmar, C., Hosny, A., Aucoin, N., Narayan, V., Beets-Tan, R. G. H., Fillon-Robin, J. C., Pieper, S., Aerts, H. J. W. L. (2017). Computational Radiomics System to Decode the Radiographic Phenotype. Cancer Research, 77(21), e104–e107. https://doi.org/10.1158/0008-5472.CAN-17-0339 <https://doi.org/10.1158/0008-5472.CAN-17-0339>_'''
import radiomics
from radiomics import featureextractor
import csv
import SimpleITK as sitk
import numpy as np

'''
Este script de Python automatiza la extracción de características radiómicas de imágenes médicas (archivos NRRD)
y sus correspondientes máscaras de segmentación, utilizando la librería PyRadiomics. Procesa un rango específico
de carpetas de datos DICOM, adapta las máscaras 2D a 3D si es necesario, y guarda todas las características
extraídas en un archivo CSV.
'''

folders=[]
for i in range(494, 503):
    folders.append(os.path.join("dicom_data",str(i)))


output_csv = 'radiomics_features_otsu3.csv'


params_file = os.path.abspath(os.path.join('params', 'Params.yaml'))


extractor = featureextractor.RadiomicsFeatureExtractor(params_file)


file_exists = os.path.exists(output_csv)

with open(output_csv, mode='a', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=';')
    
    header_written = file_exists  

    for folder in folders:
        print(folder)
        image_path = os.path.join(folder, 'nrrdFile.nrrd')
        mask_path = os.path.join(folder, 'segmentation.nrrd')
        
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Advertencia: Archivos no encontrados en la carpeta {folder}")
            continue
        
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)

        
        image_array = sitk.GetArrayFromImage(image)
        mask_array = sitk.GetArrayFromImage(mask)
        
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Advertencia: Archivos no encontrados en la carpeta {folder}")
            continue
        print(f"Dimensiones de la imagen: {image.GetDimension()}")
        print(f"Dimensiones de la máscara: {mask.GetDimension()}")
        mask_3d = np.stack([mask_array] * image_array.shape[0], axis=0)
        mask_3d_sitk = sitk.GetImageFromArray(mask_3d)
        mask_3d_sitk.CopyInformation(image)
        new_mask_path = mask_path.replace(".nrrd", "_3D.nrrd") 
        sitk.WriteImage(mask_3d_sitk, new_mask_path)

        

        
        feature_vector = extractor.execute(image_path, new_mask_path)

        
        if not header_written:
            header = ['Folder'] + list(feature_vector.keys()) 
            csv_writer.writerow(header)
            header_written = True
            print(list(feature_vector.keys()))

       
        row = [folder] + list(feature_vector.values())
        print(row)
        csv_writer.writerow(row)
        print(row)

print(f"Extracción completada. Resultados guardados en '{output_csv}'.")
