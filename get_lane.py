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

def mask_lane(image, lines):
    ret = np.zeros_like(image)
    if lines is None:
        return image
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(ret, (x1, y1), (x2, y2), (255, 255, 255), 1)
    return ret

def get_lane(img : np.array, size : np.array, vertices : np.array):
    # img : 입력 이미지, size 세로, 가로 길이, vertices : roi 꼭짓점 4개 : [왼쪽 아래, 오른쪽 아래, 오른쪽 위, 왼쪽 위]
    img = cv.resize(img, size)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blur, 50, 110)
    roi_edges = get_trapezoidal_roi(edges, vertices)
    lines = cv.HoughLinesP(roi_edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)
    masked_by_lane = mask_lane(img, lines)
    return masked_by_lane

# =============================== 데이터 불러오기 ====================================#
if __name__ == '__main__':
    file = 'imgs/drive.mp4'
    cap = cv.VideoCapture(file)
    Nframe = 0  # frame 수

    # ================================== 메인 루틴 =====================================#

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:  # 비디오 프레임을 읽기 성공했으면 진행
            frame = cv.resize(frame, (1000, 562))
        else:
            break

        Nframe += 1
        height, width = np.array([562, 1000])
        vertices = np.array([
            [0, height * 0.9],
            [width, height * 0.9],
            [width * 0.8, height * 0.6],
            [width * 0.2, height * 0.6]
        ], np.int32)

        lane = get_lane(frame,np.array([1000, 562]), vertices)


        # 이미지 원본 + 전처리 영상 출력하기
        cv.imshow('lanes', lane)  # 차선 검출 결과

        if cv.waitKey(1) & 0xff == ord('q'):  # 'q'누르면 영상 종료
            break

    print("Number of Frame: ", Nframe)  # 영상의 frame 수 출력

    cap.release()
    cv.destroyAllWindows()
