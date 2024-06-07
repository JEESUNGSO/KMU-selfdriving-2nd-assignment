import cv2 as cv
import numpy as np


def get_lane(img : np.array, threshold1, threshold2):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blur, threshold1, threshold2)


    return edges
