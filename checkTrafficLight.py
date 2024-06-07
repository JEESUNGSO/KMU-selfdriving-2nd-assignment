import cv2
import numpy as np

def check_traffic_light(image, x, y, w, h, threshold):
    if len(image.shape) == 1:  # if image is not loaded then wait(False)
        return False

    roi = image[y:y+h, x:x+w]

    green_hist = get_histogram(roi)
    num_of_green = np.count_nonzero(green_hist)
    if num_of_green > threshold:
        return True  # 출발!
    else:
        return False  # 기다림


def get_histogram(img: np.array):
    green_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    return green_hist
