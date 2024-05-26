import cv2
import numpy as np
from birds_eye_view import bird_eye_view

def get_start_window(mask_img, ratio): # mask_img: 마스크 이미지, ratio: 초기 윈도우를 구하기 위한 (최대 탐색 높이/ 전체 높이)
    # 윈도우 제일 아래 시작 지점 구하기
    # 시작 지점을 구하기 위한 마스크 생성
    mask = np.zeros_like(mask_img)
    mask[int(mask.shape[0]*(1 - ratio)):, :] = 1
    # 마스크 적용
    mask_img = mask_img * mask
    # 세로축으로 마스크 히스토그램 왼쪽 오른쪽 구하기
    mask_img_hist = np.sum(mask_img, axis=0)/255
    mask_img_hist_L = mask_img_hist.copy()
    mask_img_hist_M = mask_img_hist.copy()
    mask_img_hist_R = mask_img_hist.copy()
    mask_img_hist_R[:(mask_img_hist.shape[0]//3)*2] = 0
    mask_img_hist_M[:mask_img_hist.shape[0]//3] = 0
    mask_img_hist_M[(mask_img_hist.shape[0]//3)*2:] = 0
    mask_img_hist_L[mask_img_hist.shape[0]//3:] = 0

    return np.argmax(mask_img_hist_L), np.argmax(mask_img_hist_M), np.argmax(mask_img_hist_R)

def get_rect(y, x, w, h): # y: 중심 y좌표, x: 중심x 좌표, w: 넓이, h: 높이
    # 좌측 상단 점와 우측 하단 점을 리턴(이미지 좌표계 기준)
    return (y - h//2, x - w//2), (y + h//2, x + w//2) # 좌상단, 우하단 (y ,x)

def draw_points(img, pts):
    for pt in pts:
        cv2.circle(img, (pt[1], pt[0]), 1, 200, 2)

def draw_rect(img, pts, w, h):
    for pt in pts:
        cv2.rectangle(img, (pt[1]-max([w//2, 0]), pt[0]-max([h//2, 0])), (pt[1]+min([w//2, img.shape[1]]), pt[0]+min([h//2, img.shape[0]])), 100, 2)

def get_middle_point(mask_img, tl, br):
    window = mask_img[tl[0]:br[0], tl[1]:br[1]]
    nonzero_x = np.nonzero(window)[1]
    if len(nonzero_x) > 0: # 윈도우에 차선이 존재할 떄
        mean_x = np.mean(nonzero_x)
        return int(tl[1] + mean_x) # x 중심 좌표
    else: # 윈도우에 차선이 존재하지 않을 때
        return int((tl[1] + br[1]) / 2)

# mask_img: 전처리를 통해 검출된 차선 + ROI, window_num: 차선당 윈도우 개수
# window_width: 윈도우 폭, ratio 처음 윈도우 찾는 최고 높이 비율
def get_windows(mask_img, window_num, window_width, ratio):
    # 창 높이 구하기
    window_height = mask_img.shape[0]//window_num

    # 윈도우 중심점들 리스트
    R_middle_points = []
    M_middle_points = []
    L_middle_points = []

    # 제일 아래 시작 윈도우 중심점 얻기
    base_L, base_M, base_R = get_start_window(mask_img, ratio)
    # 제일 아래 윈도우 x좌표를 현재 윈도우 x좌표로 설정
    current_L = base_L
    current_M = base_M
    current_R = base_R
    # 바로 아래 윈도우 중심좌표로 다음 윈도우 중심 좌표 구하기
    for window in range(window_num):
        # 현재 윈도우 y 인덱스
        y_ = mask_img.shape[0] - window_height // 2 - window_height * window
        # 현재 윈도우 중심점 리스트에 추가, (y, x)로 좌표(인덱스)를 저장
        R_middle_points.append((y_, current_R))
        M_middle_points.append((y_, current_M))
        L_middle_points.append((y_, current_L))
        # 제일 아래 윈도우의 중심점을 중심으로 하는 윈도우를 위에 생성
        # 왼쪽 위, 오른쪽 아래 점 구하기
        R_tl, R_br = get_rect(y_, current_R, window_width, window_height)
        M_tl, M_br = get_rect(y_, current_M, window_width, window_height)
        L_tl, L_br = get_rect(y_, current_L, window_width, window_height)
        # 다음 중심점 구하기
        current_R = get_middle_point(mask_img, R_tl, R_br)
        current_M = get_middle_point(mask_img, M_tl, M_br)
        current_L = get_middle_point(mask_img, L_tl, L_br)


    # 시각화
    draw_points(mask_img, R_middle_points + L_middle_points + M_middle_points)
    draw_rect(mask_img, R_middle_points + L_middle_points + M_middle_points, window_width, window_height)

    return L_middle_points, M_middle_points, R_middle_points


# 테스트용 실행
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # 이미지 불러오기
    mask_img = cv2.imread('imgs/mask3.png')
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

    # 이미지 birds eye view로 변환
    mask_img = bird_eye_view(mask_img, 0, int(mask_img.shape[0]/1.5), 444, 120)


    # 슬라이딩 윈도우
    result = get_windows(mask_img, 20, 50,0.1)
    
    # 출력
    cv2.imshow('mask_img', mask_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()