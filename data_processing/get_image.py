import cv2

video_file = "data.mp4"  # 동영상 파일 경로
save_dir = 'driving_images/'

cap = cv2.VideoCapture(video_file)

cnt = 0
if cap.isOpened():
    while True:
        ret, img = cap.read()
        if ret:
            if cnt % 3 == 0:
                img = cv2.resize(img, (160, 120))
                cv2.imwrite(f"{save_dir}img_{int(cnt/3)}.jpg", img)

                cv2.imshow(video_file, img)
                cv2.waitKey(1)
        else:
            break
        cnt += 1
else:
    print("can't open video.")
cap.release()
cv2.destroyAllWindows()