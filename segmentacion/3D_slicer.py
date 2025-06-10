'''
This script, designed to be run in the 3D Slicer Python console, 
automates the segmentation of a DICOM volume. It loads a 
specified DICOM file, creates a new segmentation, applies a 
thresholding effect to isolate high-intensity regions, refines 
the segmentation with margin operations (shrink and grow) and 
smoothing, and finally saves the resulting segmentation as an NRRD 
file. It specifically aims to segment a "HandMask" by isolating 
the largest connected component after initial thresholding.
'''
i=500                                                                                     
min_threshold =24670.80                       

dicom_path = f"C:\\Users\\JUAN\\Documents\\GitHub\\temp\\dicom_data\\{str(i)}\\dicomFile.dcm"
volumeNode = slicer.util.loadVolume(dicom_path)

if not volumeNode:
    print(f"Error: No se pudo cargar el volumen {dicom_path}")

# Crear nodo de segmentación
segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
segmentationNode.CreateDefaultDisplayNodes()
segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)

# Crear editor de segmentos
segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
segmentEditorWidget.setSegmentationNode(segmentationNode)
segmentEditorWidget.setSourceVolumeNode(volumeNode)
segmentEditorNode.SetOverwriteMode(slicer.vtkMRMLSegmentEditorNode.OverwriteNone)

# Crear segmento
segmentName = "HandMask"
segmentationNode.GetSegmentation().AddEmptySegment(segmentName)
segmentEditorNode.SetSelectedSegmentID(segmentName)  # Seleccionar el segmento

# ✅ Verificar que el segmento fue agregado correctamente
segmentID = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentName)
if not segmentID:
    print(f"Error: No se pudo encontrar el segmento {segmentName}")


num_islands=0

cicle=0

imageData = volumeNode.GetImageData()

# Verificar si la imagen es válida
if imageData:
    minIntensity, maxIntensity = imageData.GetScalarRange()
    print(f"🔍 Intensidad mínima: {minIntensity}, Intensidad máxima: {maxIntensity}")
else:
    print("⚠️ No se pudo obtener la imagen del volumen.")


print(f"🔍 Iteración {cicle+1}: Aplicando umbral mínimo de {min_threshold}")
cicle+=1
segmentEditorWidget.setActiveEffectByName("Threshold")
effect1 = segmentEditorWidget.activeEffect()
effect1.setParameter("MinimumThreshold", min_threshold)
effect1.setParameter("MaximumThreshold", maxIntensity)
effect1.self().onApply()

    

# Aplicar reducción de margen
segmentEditorWidget.setActiveEffectByName("Margin")
effect2 = segmentEditorWidget.activeEffect()
effect2.setParameter("MarginSizeMm", str(-1.0))  # Reducir 1 mm
effect2.self().onApply()

segmentation = segmentationNode.GetSegmentation()
segmentIDs = [segmentation.GetNthSegmentID(i) for i in range(segmentation.GetNumberOfSegments())]

# Buscar el segmento con el mayor volumen
max_volume = 0
largest_segment_id = None

segmentEditorWidget.setActiveEffectByName("Islands")
effect4 = segmentEditorWidget.activeEffect()
effect4.setParameter("Operation", "KEEP_LARGEST_ISLAND")
effect4.self().onApply()

    # Aplicar expansión de margen
segmentEditorWidget.setActiveEffectByName("Margin")
effect3 = segmentEditorWidget.activeEffect()
effect3.setParameter("MarginSizeMm", str(1.0))  # Expandir 1 mm
effect3.self().onApply()

# Aplicar suavizado

segmentEditorWidget.setActiveEffectByName("Smoothing")
effect5 = segmentEditorWidget.activeEffect()
effect5.setParameter("ApplyToAllVisibleSegments", 0)
effect5.setParameter("SmoothingMethod", "GAUSSIAN")
effect5.setParameter("GaussianStandardDeviationMm", 2.0)
effect5.self().onApply()



slicer.mrmlScene.RemoveNode(segmentEditorNode)

# Guardar el resultado
output_nrrd_path = f"C:\\Users\\JUAN\\Documents\\GitHub\\temp\\dicom_data\\{i}\\segmentation_3dslicer_2.nrrd"
slicer.util.saveNode(segmentationNode, output_nrrd_path)

print(f"✅ Segmentación completada y guardada en {output_nrrd_path}")
#slicer.mrmlScene.RemoveNode(segmentationNode)