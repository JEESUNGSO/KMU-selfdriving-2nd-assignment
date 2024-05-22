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

def compare_img(img1: np.array, img2: np.array) :
    hist1 = get_histogram(img1)
    hist2 = get_histogram(img2)

    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT) / np.sum(hist1)

def get_histogram(img: np.array):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # RGB 형태의 이미지를 HSV(색상, 채도, 명도) 형태로 바꾸어 줌
    # HSV는 조명 변화에 강하고 색상 구분이 용이하기 때문

    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # 색조(Hue)와 채도(Saturation)에 대해 히스토그램 생성, 
    # 전체 이미지에 대해 시행, 
    # 색조에 대해서는 180개의 빈, 채도에 대해서는 256개의 빈을 가지도록
    # 빈 개수가 많을 수록 더 정밀한 분포를 얻음!
    # 색조에 대해서는 0 ~ 180범위의 채널
    # 명도에 대해서는 0 ~ 255 범위의 채널
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


if __name__ == '__main__':
    img = cv2.imread('img.jpg', cv2.IMREAD_COLOR)
    roi = get_roi(img, 35, 35, 256, 32)
    # roi test
    cv2.imshow('img', img)
    cv2.imshow('roi', roi)

    # compare img test
    roi1 = get_roi(img, 32, 32, 256, 32)
    roi2 = get_roi(img, 32, 128 , 256, 32)
    cv2.imshow('ro11', roi1)
    cv2.imshow('roi2', roi2)
    print("compare roi and roi1", compare_img(roi, roi1))
    print("compare roi and roi2", compare_img(roi, roi2))
    cv2.waitKey()