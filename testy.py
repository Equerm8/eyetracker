import cv2 as cv

import cv2 as cv
import dlib
import image_processing as img_proc
import utils
import numpy as np

#TODO: odpornosc na bledy, tj. gdy nie ma twarzy to nie wywala
#TODO: zabezpieczenie countera przed overflow
#TODO: zwracanie koordynatow

class Pupil():

    def __init__(self, rows:int = 3, cols:int = 3, k:int = 4, binInt:int = 6) -> None:
        self.rows = rows
        self.cols = cols
        self.k = k
        self.binInt = binInt
        self.imgEye = None
        self.xEyeGaze = None
        self.yEyeGaze = None

        self.xEyeGaze2Train = 0
        self.yEyeGaze2Train = 0

        self._hogFaceDetector = dlib.get_frontal_face_detector()
        self._dlib_facelandmark = dlib.shape_predictor("C:/Users/Equer/Desktop/Uczelnia/Projekty/SKN SKORP/Eye Tracker/Skorp Tracker/demo/SkorpTracker ver-03/shape_predictor_68_face_landmarks.dat")
        self._counter = [[1 for _ in range(cols)] for _ in range(rows)]

        # TEST
        self.gray = None
        self.bin = None
        
    def __str__(self) -> str:
        return f"Gaze coordinates: [{self.xGaze}, {self.yGaze}]"

    def reset_counter(self) -> None:
        self._counter = [[1 for _ in range(self.cols)] for _ in range(self.rows)]

    def get_coordinates(self, img: np.ndarray) -> None:
        
        #! BRANA JEST POD UWAGE TYLKO JEDNA ZRENICA
        self.imgEye = img
        
        imgSegmL = img_proc.color_segmentation(img)

        imgGrey = cv.cvtColor(imgSegmL, cv.COLOR_BGR2GRAY)
        _, binarized = cv.threshold(imgGrey, self.binInt, 255, cv.THRESH_BINARY_INV)
        self.gray = imgGrey
        self.bin = binarized
        firstLoop = True
        # Dividing an image
        if firstLoop:
            imgs = img_proc.divide_img(binarized, self.rows, self.cols)
            firstLoop = False
            searches = [[False for _ in range(self.cols)] for _ in range(self.rows)]
        allTrue = False

        while(allTrue == False):
            i, j = utils.find_max_arr_val(self.cols, self.rows, searches, self._counter)
            contours, hierarchy = cv.findContours(imgs[i][j], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            for contour in contours:
                moments = cv.moments(contour, True)
                if moments["m00"] != 0:
                    height, width = imgs[i][j].shape
                    self.xEyeGaze2Train = int(moments["m10"] / moments["m00"]) + width*j
                    self.yEyeGaze2Train = int(moments["m01"] / moments["m00"]) + height*i
                    self._counter[i][j] += 1
                    return True
                else:
                    continue
            allTrue = utils.is_end(self.rows, self.cols, searches)


        
        return True
    
        self.xEyeGaze = None
        self.yEyeGaze = None
        self.imgEye = None

imgs = []
#! load imgs


def preprocess_img(img, targetSize = (128, 128)):
    resized_img = cv.resize(img, targetSize)
    imgSegmL = img_proc.color_segmentation(resized_img)
    imgGrey = cv.cvtColor(imgSegmL, cv.COLOR_BGR2GRAY)
    normalized_img = imgGrey / 255.0
    return normalized_img

import time
getCoorT = 0
pupil = Pupil(3, 3)

start_time = time.time()
for img in imgs:
    pupil.get_coordinates(img)
end_time = time.time()
getCoorT = end_time - start_time
print("Czas opt.", getCoorT)

start_time = time.time()
for i, img in enumerate(imgs):
    imgSegmL = img_proc.color_segmentation(img)
    imgGrey = cv.cvtColor(imgSegmL, cv.COLOR_BGR2GRAY)
    _, binarized = cv.threshold(imgGrey, 6, 255, cv.THRESH_BINARY_INV)
    contours1, hierarchy = cv.findContours(binarized, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contoursCoor1 = []
    for contour in contours1:
        moments1 = cv.moments(contour, True)
        if moments1["m00"] != 0:
            cx1 = int(moments1["m10"] / moments1["m00"])
            cy1 = int(moments1["m01"] / moments1["m00"])
            break
        else:
            continue
end_time = time.time()

getCoorT = end_time - start_time
print("Czas bez opt.", getCoorT)


from keras import models
model = models.load_model("image_classifier_7.keras")
import math

start_time = time.time()
for i, img in enumerate(imgs):
    old_x, old_y, _ = img.shape

    img_to_predict = preprocess_img(img)
    new_x, new_y = img_to_predict.shape
    cols_ext = old_x/new_x
    rows_ext = old_y/new_y

    img_to_predict = np.stack((img_to_predict, img_to_predict, img_to_predict), axis=-1)  # Kopiowanie szarego kanału
    prediciton = model.predict(np.array([img_to_predict]), verbose = 0)
    
    x = math.floor(prediciton[0][1] * cols_ext)
    y = math.floor(prediciton[0][0] * rows_ext)
    img_cont = cv.circle(img, (y, x), 2, (247, 0, 255), 2)

end_time = time.time()
getCoorT = end_time - start_time
print("Czas CNN", getCoorT)