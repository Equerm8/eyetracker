import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import image_processing as ip

# pobranie zdjec
input_folder = r"CNN/training_photos"
images = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))

for i, img in enumerate(images):
    image_path = os.path.join(r"CNN/training_photos", img)
    name = images[i]
    x, y = ip.parse_label(name)
    plt.scatter(x, y, color='blue')  # Rysowanie punktu
    plt.text(x + 0.1, y + 0.1, f"{x},{y}", fontsize=9)  # Dodanie etykiety punktu
    if i == 2000:break


# Dodanie szczegółów wykresu

# Dodanie szczegółów wykresu
plt.title("Punkty z listy krotek")
plt.xlabel("X")
plt.ylabel("Y")
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()
input("Naciśnij Enter, aby zamknąć wykres...")