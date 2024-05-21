import cv2
import numpy as np


# x, y를 기준으로 아래로 h 만큼, 우측으로 w만큼의 영역을 선택하여 리턴
def get_roi(img: np.array, x: int, y: int, w: int, h: int) -> np.array:
    roi = img[y:y + h, x:x + w]
    return roi

def check_traffic_light(greenImg: np.array, yellowImg: np.array, redImg: np.array, roi: np.array) -> int:
    g_sim = compare_img(greenImg, roi)
    y_sim = compare_img(yellowImg, roi)
    r_sim = compare_img(redImg, roi)
    return np.argmax([g_sim, y_sim, r_sim]) # 초록색에 가장 비슷하면 0, 노란색 1, 빨간색 2 리턴

def compare_img(img1: np.array, img2: np.array) -> float:
    hist1 = get_histogram(img1)
    hist2 = get_histogram(img2)
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT) # 히스토그램 교차를 통해 비교 (0 ~ 1)

def get_histogram(img: np.array):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # 일단 임의값으로 넣어 놨음 수정 필요!
    return hist
