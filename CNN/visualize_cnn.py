import visualkeras
import matplotlib.pyplot as plt
#from keras import models


#model = models.load_model("image_classifier_7.keras")
# from PIL import ImageFont

# font = ImageFont.truetype("times.ttf", 14)  # using comic sans is strictly prohibited!
# visualkeras.layered_view(model, to_file='output1.png', legend=True, font=font) # write to disk
# visualkeras.layered_view(model, to_file='output1.png', legend=True, font=font).show() # write and show
# visualkeras.layered_view(model).show() # display using your system viewer

#print(model.summary())
import pickle
with open('training_history_img_class_7.pkl', 'rb') as file:
    loaded_history = pickle.load(file)

epochs = list(range(1, 21))  # Oś x od 1 do 20

# for x in loaded_history:
#     if x == "val_accuracy":
        
#         # Tworzenie wykresu
#         plt.figure(figsize=(8, 5))  # Opcjonalne: rozmiar wykresu
#         y = loaded_history[x]
#         for i, el in enumerate(y):
#             y[i] = el * 100
#         plt.scatter(epochs, y, color='blue', label='Dokładność', s=20)  # Punkty na wykresie

#         # Nazwy osi
#         plt.xlabel('Epoka')
#         plt.ylabel('Dokładność [%]')

#         # Automatyczne dopasowanie zakresu osi y
#         plt.ylim(60, max(loaded_history[x]) + 5)
#         plt.xticks(ticks=epochs)
#         # Opcjonalne: dodanie siatki i tytułu
#         plt.grid(True, linestyle='--', alpha=0.6)
#         plt.title('Dokładność na przestrzeni kolejnych epok.')

#         # Wyświetlenie wykresu
#         plt.show()

for x in loaded_history:
    if x == "val_root_mean_squared_error":
        
        # Tworzenie wykresu
        plt.figure(figsize=(8, 5))  # Opcjonalne: rozmiar wykresu
        y = loaded_history[x]
        plt.scatter(epochs, y, color='red', label='Pierwiastek z błędu średniokwadratowego', s=20)  # Punkty na wykresie

        # Nazwy osi
        plt.xlabel('Epoka')
        plt.ylabel('Pierwiastek z błędu średniokwadratowego [px]')

        # Automatyczne dopasowanie zakresu osi y
        plt.ylim(0, max(loaded_history[x]) + 1)
        plt.xticks(ticks=epochs)
        # Opcjonalne: dodanie siatki i tytułu
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.title('RMSE na przestrzeni kolejnych epok.')

        # Wyświetlenie wykresu
        plt.show()