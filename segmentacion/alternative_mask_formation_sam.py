import os
import glob
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np


'''
This Python script provides a graphical user interface (GUI) built with 
tkinter that allows users to visualize and manage image files 
(specifically, .png masks and other common image formats) within a 
predefined folder. Users can select multiple images, see a live "mixed" 
(bitwise OR) preview of the unselected images, and then delete the 
selected images from the disk.
'''


# Folder path where the images are stored
FOLDER_PATH = r""

def load_images():
    images = []
    for file in os.listdir(FOLDER_PATH):
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
            images.append(os.path.join(FOLDER_PATH, file))
    return images

def mix_images(selected_images):
    mask_paths = glob.glob(FOLDER_PATH + "/*.png")
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

def show_images():
    images = load_images()
    if not images:
        messagebox.showinfo("Información", "No se encontraron imágenes en la carpeta predefinida.")
        return
    
    def toggle_selection(index):
        if index in selected_images:
            selected_images.remove(index)
            buttons[index].config(relief=tk.RAISED)
        else:
            selected_images.add(index)
            buttons[index].config(relief=tk.SUNKEN)
        show_mixed_image()
    
    def show_mixed_image():
        mixed_img = mix_images({images[i] for i in selected_images})
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
            root.destroy()
            main_root.destroy()
    
    root = tk.Toplevel()
    root.title("Seleccionar imágenes a eliminar")
    root.geometry("800x500")
    root.protocol("WM_DELETE_WINDOW", lambda: [root.destroy(), main_root.destroy()])
    
    frame = tk.Frame(root)
    frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    canvas = tk.Canvas(frame)
    scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas)
    
    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    
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
    
    mixed_img = mix_images({})
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
