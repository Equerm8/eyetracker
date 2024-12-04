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
        # type your directory with dlib_facelandmarks
        dir = r"YOUR\dir"
        self._dlib_facelandmark = dlib.shape_predictor(dir)
        self._counter = [[1 for _ in range(cols)] for _ in range(rows)]

        # TEST
        self.gray = None
        self.bin = None
        
    def __str__(self) -> str:
        return f"Gaze coordinates: [{self.xGaze}, {self.yGaze}]"

    def reset_counter(self) -> None:
        self._counter = [[1 for _ in range(self.cols)] for _ in range(self.rows)]

    def get_coordinates(self, img: np.ndarray) -> None:
        imgGrey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = self._hogFaceDetector(imgGrey) # look for faces
        for face in faces:
            #! BRANA JEST POD UWAGE TYLKO JEDNA ZRENICA
            faceLandmarks = self._dlib_facelandmark(imgGrey, face)
            x1L = faceLandmarks.part(36).x
            y1L = faceLandmarks.part(37).y
            x2L = faceLandmarks.part(39).x
            y2L = faceLandmarks.part(41).y
            
            imgCroppedL = img[y1L:y2L, x1L:x2L]
            imgCroppedL = img_proc.resize_img(imgCroppedL, self.k)
            self.imgEye = imgCroppedL
            
            imgSegmL = img_proc.color_segmentation(imgCroppedL)
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
                #print(i,j)
                contours, hierarchy = cv.findContours(imgs[i][j], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                for contour in contours:
                    moments = cv.moments(contour, True)
                    if moments["m00"] != 0:
                        height, width = imgs[i][j].shape
                        self.xEyeGaze2Train = int(moments["m10"] / moments["m00"]) + width*j
                        self.yEyeGaze2Train = int(moments["m01"] / moments["m00"]) + height*i
                        self._counter[i][j] += 1
                        #print(self._counter)
                        return True
                    else:
                        continue
                allTrue = utils.is_end(self.rows, self.cols, searches)

        self.xEyeGaze = None
        self.yEyeGaze = None
        self.imgEye = None
        return False
    
        
