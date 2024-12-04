import cv2 as cv
import pupil_tracker as pt
from keras import models
import tensorflow as tf
import keras
import math
from keras import models
import numpy as np

def preprocess_img(img, targetSize = (128, 128)):
    resized_img = cv.resize(img, targetSize)
    normalized_img = resized_img / 255.0
    return normalized_img


@keras.saving.register_keras_serializable()
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

gaze = pt.Pupil(3, 3, 3, 10)
model = models.load_model("image_classifier_7.keras")

cap = cv.VideoCapture('test_eyetracker3.mp4')
#cap = cv.VideoCapture(1)

i = 0

import time
while(cap.isOpened()):
#while(True):
    _, frame = cap.read()

    if gaze.get_coordinates(frame):

        #contours algorithm
        imgEye2 = gaze.imgEye
        imgCpy1 = imgEye2.copy()
        imgCpy2 = imgEye2.copy()
        img_cont = cv.circle(imgEye2, (gaze.xEyeGaze2Train, gaze.yEyeGaze2Train), 1, (247, 0, 255), 1)
        cv.imwrite(rf'val\imgs_opt3_p\{i}.jpg', img_cont)
        img_cont = cv.circle(imgEye2, (gaze.xEyeGaze2Train, gaze.yEyeGaze2Train), 1, (247, 0, 255), 2)
        cv.imwrite(rf'vis\imgs_opt3_p\{i}.jpg', img_cont)
        cv.imshow('Cont', img_cont)
        

        #contours without opt
        contours1, hierarchy = cv.findContours(gaze.bin, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        contoursCoor1 = []
        for contour in contours1:
            moments1 = cv.moments(contour, True)
            if moments1["m00"] != 0:
                cx1 = int(moments1["m10"] / moments1["m00"])
                cy1 = int(moments1["m01"] / moments1["m00"])
                contoursCoor1.append((cx1, cy1))
            else:
                continue
        if len(contoursCoor1) > 0:
            i7a = cv.circle(imgCpy1, (contoursCoor1[0][0], contoursCoor1[0][1]), 1, (0, 255, 0), 1)
            cv.imwrite(rf'val\imgs_bez_opt3_p\{i}.jpg', i7a)
            i7a = cv.circle(imgCpy1, (contoursCoor1[0][0], contoursCoor1[0][1]), 1, (0, 255, 0), 2)
            cv.imwrite(rf'vis\imgs_bez_opt3_p\{i}.jpg', i7a)
            cv.imshow('h', i7a)
        

        # CNN
        old_x, old_y, _ = imgEye2.shape
        img_to_predict = preprocess_img(gaze.gray)
        new_x, new_y = img_to_predict.shape
        cols_ext = old_x/new_x
        rows_ext = old_y/new_y

        img_to_predict = np.stack((img_to_predict, img_to_predict, img_to_predict), axis=-1)  # Kopiowanie szarego kanału
        prediciton = model.predict(np.array([img_to_predict]), verbose = 0)
        
        x = math.floor(prediciton[0][1] * cols_ext)
        y = math.floor(prediciton[0][0] * rows_ext)
        img_cnn = cv.circle(imgCpy2, (y, x), 1, (0, 0, 255), 1)
        cv.imwrite(rf'val\imgs_cnn3_p\{i}.jpg', img_cnn)
        img_cnn = cv.circle(imgCpy2, (y, x), 1, (0, 0, 255), 2)
        cv.imwrite(rf'vis\imgs_cnn3_p\{i}.jpg', img_cnn)
        cv.imshow('CNN', img_cnn)

        

    if cv.waitKey(1)>=0 & 0xFF == ord('q'):
        break
    
    i += 1

cap.release()
cv.destroyAllWindows()