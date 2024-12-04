import cv2 as cv
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import image_processing as ip

# pobranie zdjec
input_folder = r"CNN/testing_photos_normal"
output_folder = r"CNN/testing_photos_gray"
images = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
for i, img in enumerate(images):
    image_path = os.path.join(r"CNN/testing_photos_normal", img)
    image = cv.imread(image_path)
    name = images[i]

    # resize
    org_size_rows, org_size_cols, _ = image.shape
    image = ip.resize_preprocess_img(image, targetSize=(128, 128))
    new_size_rows, new_size_cols, _ = image.shape
    
    # new coordinates
    cols_ext = new_size_cols/org_size_cols
    rows_ext = new_size_rows/org_size_rows
    old_x, old_y = ip.parse_label(name)
    new_x = int(old_x * cols_ext)
    new_y = int(old_y * rows_ext)
    name = f"({i}, {new_x}, {new_y}).jpg"

    # color segmentation
    image = ip.color_segmentation(image)

    # rgb2gray
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # zapis do pliku
    cv.imwrite(rf"{output_folder}/{name}", image)