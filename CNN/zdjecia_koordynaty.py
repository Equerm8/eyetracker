import cv2
import os

# Ścieżka do folderu źródłowego z obrazami
input_folder = "training_photos"
# Ścieżka do folderu docelowego, gdzie zapisane będą obrazy po kliknięciu
output_folder = "testing_photos"

# Tworzenie folderu docelowego, jeśli nie istnieje
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Inicjalizacja indeksu obrazu
image_index = 0

# Funkcja, która obsługuje kliknięcia na obrazie
def on_mouse_click(event, x, y, flags, param):
    global image_index, output_folder, image
    if event == cv2.EVENT_LBUTTONDOWN:
        # Zapisz obraz w folderze docelowym z nazwą w formacie (index, x, y).jpg
        output_filename = f"{output_folder}/({image_index}, {x}, {y}).jpg"
        cv2.imwrite(output_filename, image)
        print(f"Obraz zapisany jako {output_filename}")
        # Zwiększ indeks obrazu i zamknij aktualne okno
        image_index += 1
        cv2.destroyAllWindows()

# Pobranie listy plików z folderu źródłowego
images = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Pętla po wszystkich obrazach w folderze
for image_file in images:
    # Wczytaj obraz
    image_path = os.path.join(input_folder, image_file)
    print(image_path)
    image = cv2.imread(image_path)

    # Sprawdzenie, czy obraz został poprawnie wczytany
    if image is None:
        print(f"Nie udało się wczytać obrazu: {image_file}")
        continue

    # Wyświetlenie obrazu
    cv2.imshow("Kliknij na obraz", image)
    # Przypisanie funkcji kliknięcia do okna
    cv2.setMouseCallback("Kliknij na obraz", on_mouse_click)

    # Czekaj na kliknięcie, zamknięcie okna spowoduje przejście do kolejnego obrazu
    cv2.waitKey(0)

# Po zakończeniu przetwarzania obrazów zamknij wszystkie okna
cv2.destroyAllWindows()
