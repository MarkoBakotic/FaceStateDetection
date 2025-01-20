import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

# Funkcija koja mijenja izvor boje (kanal slike)
def promijeni_kanal(kanal):
    global original_image, image_label
    
    # Pretvori originalnu sliku u NumPy array
    image_array = np.array(original_image)
    
    # Prazni ostale kanale ovisno o odabranom kanalu
    if kanal == 'R':  # Crveni kanal
        image_array[:, :, 1] = 0  # Zeleni kanal na 0
        image_array[:, :, 2] = 0  # Plavi kanal na 0
    elif kanal == 'G':  # Zeleni kanal
        image_array[:, :, 0] = 0  # Crveni kanal na 0
        image_array[:, :, 2] = 0  # Plavi kanal na 0
    elif kanal == 'B':  # Plavi kanal
        image_array[:, :, 0] = 0  # Crveni kanal na 0
        image_array[:, :, 1] = 0  # Zeleni kanal na 0

    # Ponovno stvori sliku iz NumPy array-a
    new_image = Image.fromarray(image_array)
    new_image = ImageTk.PhotoImage(new_image)
    
    # Ažuriraj labelu s novom slikom
    image_label.config(image=new_image)
    image_label.image = new_image

# Funkcija za učitavanje slike
def učitaj_sliku():
    global original_image
    file_path = filedialog.askopenfilename(title="Odaberite sliku", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    
    if file_path:
        # Otvori sliku
        original_image = Image.open(file_path)
        original_image = original_image.resize((300, 300))  # Preporučujemo da slika bude manje dimenzije za bolju izvedbu
        original_image = ImageTk.PhotoImage(original_image)
        
        # Prikaz slike na labeli
        image_label.config(image=original_image)
        image_label.image = original_image

# Kreiraj glavni prozor
root = tk.Tk()
root.title("RGB Kanali Slike")
root.geometry("400x500")

# Labela za prikaz slike
image_label = tk.Label(root)
image_label.pack(pady=20)

# Gumbi za promjenu kanala boje
button_r = tk.Button(root, text="R (Crveni)", command=lambda: promijeni_kanal('R'))
button_r.pack(side="left", padx=10)

button_g = tk.Button(root, text="G (Zeleni)", command=lambda: promijeni_kanal('G'))
button_g.pack(side="left", padx=10)

button_b = tk.Button(root, text="B (Plavi)", command=lambda: promijeni_kanal('B'))
button_b.pack(side="left", padx=10)

# Gumb za učitavanje slike
load_button = tk.Button(root, text="Učitaj sliku", command=učitaj_sliku)
load_button.pack(pady=10)

# Pokreni glavnu petlju za GUI aplikaciju
root.mainloop()
