import numpy as np
from keras import models, layers, metrics, regularizers, optimizers

import download_imgs

data_train = np.array(download_imgs.training_img_arr)
label_train = np.array(download_imgs.training_label_arr)
data_test = np.array(download_imgs.testing_img_arr)
label_test = np.array(download_imgs.testing_label_arr)

model = models.Sequential([])
model.add(layers.Conv2D(16, (5, 5), activation='relu', padding='same', input_shape=(128,128,3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(layers.Conv2D(32, (3,3), activation='relu',  padding='same'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3,3), activation='relu',  padding='same'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation='relu',  padding='same'))
model.add(layers.Conv2D(128, (3,3), activation='relu',  padding='same'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.L2(l2=0.01)))
model.add(layers.Dense(2, activation='linear'))

print(model.summary())

opt = optimizers.Adam(learning_rate=0.00001)
model.compile(optimizer=opt, metrics= [metrics.RootMeanSquaredError(), 'accuracy'], loss = 'mse')
history = model.fit(data_train, label_train, epochs=50, validation_data=(data_test, label_test), batch_size=16, verbose=2)
loss, rmse, accuracy = model.evaluate(data_test, label_test)

print(f'Loss={loss}')
print(f'accuracy={accuracy}')
print(f'rmse={rmse}')

model.save("pupil_reg_18.keras")
import pickle
with open('validation_history_img_class_18.pkl', 'wb') as file:
    pickle.dump(history.history, file)

with open('validation_history_img_class_18.pkl', 'rb') as file:
    loaded_history = pickle.load(file)
print(loaded_history)