import os  # needed navigate the system to get the input data
import radiomics
'''van Griethuysen, J. J. M., Fedorov, A., Parmar, C., Hosny, A., Aucoin, N., Narayan, V., Beets-Tan, R. G. H., Fillon-Robin, J. C., Pieper, S., Aerts, H. J. W. L. (2017). Computational Radiomics System to Decode the Radiographic Phenotype. Cancer Research, 77(21), e104–e107. https://doi.org/10.1158/0008-5472.CAN-17-0339 <https://doi.org/10.1158/0008-5472.CAN-17-0339>_'''
from radiomics import featureextractor
import csv
import SimpleITK as sitk
import numpy as np


'''
This Python script automates the extraction of radiomic features from medical 
images and their corresponding segmentation masks. It uses the PyRadiomics 
library to process a specified range of data folders, extracting various 
quantitative features and saving them into a CSV file for further analysis.
'''

folders=[]
for i in range(496, 503):
    folders.append(os.path.join("dicom_data",str(i)))

output_csv = 'radiomics_features_3dslicer.csv'


params_file = os.path.abspath(os.path.join('params', 'Params.yaml'))


extractor = featureextractor.RadiomicsFeatureExtractor(params_file)


file_exists = os.path.exists(output_csv)

with open(output_csv, mode='a', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=';')
    
    header_written = file_exists 

    for folder in folders:
        print(folder)
        image_path = os.path.join(folder, 'nrrdFile.nrrd')
        mask_path = os.path.join(folder, '3dslicer_segmentation.nrrd')

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Advertencia: Archivos no encontrados en la carpeta {folder}")
            continue
 
        feature_vector = extractor.execute(image_path, mask_path)

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
