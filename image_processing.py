import cv2 as cv
import numpy as np

def resize_img(img, k):
    dimensions = img.shape
    try:
        imgColor = cv.resize(img, (dimensions[1] * k, dimensions[0] * k))
    except: 
        return img
    
    return imgColor

def color_segmentation(img):
    R, G, B = cv.split(img)
    output1_R = cv.equalizeHist(R)
    output1_G = cv.equalizeHist(G)
    output1_B = cv.equalizeHist(B)
    imgResult = cv.merge((output1_R, output1_G, output1_B))

    return imgResult

def divide_img(img, rows, cols):
    imgs = [[False for _ in range(cols)] for _ in range(rows)]
    dimensions = img.shape
    x = dimensions[1] // cols
    y = dimensions[0] // rows

    var1x = 0; var2x = 1; var3y = 0; var4y = 1
    for i in range(0, rows):
        for j in range(0, cols):
            imgs[i][j] = img[int(y*var3y) : int(y*var4y), int(x*var1x) : int(x*var2x)]
            var1x += 1; var2x += 1
        var1x = 0; var2x = 1; var3y += 1; var4y += 1
        
    return imgs

def resize_preprocess_img(img, targetSize = (128, 128)):
    resized_img = cv.resize(img, targetSize)
    return resized_img

def parse_label(label):
    label, _ = label.split('.')
    _, first, second = label.split(',')
    first = int(first.strip())
    second = second.strip()
    second = int(second.replace(')', ''))
    
    return (first, second)
