import cv2 as cv
import numpy as np



def get_trapezoidal_roi(img, vertices):
    # Create a mask with the same dimensions as the input image
    mask = np.zeros_like(img)

    # Fill the mask with the polygon defined by "vertices"
    cv.fillPoly(mask, [vertices], 255)

    # Apply the mask to the input image
    masked_img = cv.bitwise_and(img, mask)

    return masked_img


def get_lane(img : np.array, vertices : np.array):
    # img : 입력 이미지, size 세로, 가로 길이, vertices : roi 꼭짓점 4개 : [왼쪽 아래, 오른쪽 아래, 오른쪽 위, 왼쪽 위]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blur, 50, 90)
    roi_edges = get_trapezoidal_roi(edges, vertices)


    return roi_edges

# =============================== 데이터 불러오기 ====================================#
if __name__ == '__main__':
    image = cv.imread('imgs/road_test_imgs/sunlight2.png')

    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    h, s, v = cv.split(image_hsv)

    cv.imshow('h', h)
    cv.imshow('s', s)
    cv.imshow('v', v)

    canny = cv.Canny(image, 50, 50)

    cv.imshow('canny', canny)

    height, width = image.shape[:2]
    vertices = np.array([
        [0, height * 0.9],
        [width, height * 0.9],
        [width * 0.8, height * 0.6],
        [width * 0.2, height * 0.6]
    ], np.int32)

    lane_mask = get_lane(image, vertices)


    # 출력
    cv.imshow('Lane mask', lane_mask)
    cv.waitKey(0)
    cv.destroyAllWindows()
