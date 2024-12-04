import os
import sys
import numpy as np
import cv2 as cv
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import image_processing as img_proc

img = cv.imread(r"C:\Users\Equer\Desktop\Uczelnia\Projekty\SKN SKORP\Eye Tracker\Skorp Tracker\demo\SkorpTracker ver-04\CNN\eyes_database_coordinates\(655, 56, 22).jpg")

img2r = img_proc.resize_img(img, 3)
img2 = img_proc.color_segmentation(img2r)
img2gr = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
_, img2 = cv.threshold(img2gr, 6, 255, cv.THRESH_BINARY_INV)

imgs = img_proc.divide_img(img2, 3, 3)
cv.imshow('entire', img2)

imgs2 = img_proc.divide_img(img2gr, 3, 3)
cv.imshow('entire_gray', img2gr)

for i, img in enumerate(imgs2):
    for j, im in enumerate(img):
        cv.imshow(f'{i+10}{j+10}', np.array(im))

for i, img in enumerate(imgs):
    for j, im in enumerate(img):
        cv.imshow(f'{i}{j}', np.array(im))

cv.waitKey(0)









































