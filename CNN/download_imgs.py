import cv2 as cv
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import image_processing as ip

# trening pobranie zdjec i labelow
training_path = r"CNN/training_photos_2"
imgs = os.listdir(training_path)

training_img_arr = []
training_label_arr = []

for img in imgs:
    img_path = os.path.join(training_path, img)
    unprocessed_image = cv.imread(img_path)
    if unprocessed_image is not None:
        processed_img = unprocessed_image / 255.0
        training_img_arr.append(processed_img)
        label = ip.parse_label(img)
        training_label_arr.append(label)



# uczenie pobieranie zdjeci i labelow
testing_path = r"CNN/testing_photos_gray"
imgs = os.listdir(testing_path)

testing_img_arr = []
testing_label_arr = []

for img in imgs:
    img_path = os.path.join(testing_path, img)
    unprocessed_image = cv.imread(img_path)

    if unprocessed_image is not None:
        processed_img = unprocessed_image / 255.0
        testing_img_arr.append(processed_img)

        label = ip.parse_label(img)
        testing_label_arr.append(label)