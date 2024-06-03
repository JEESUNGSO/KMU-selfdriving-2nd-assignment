import cv2


class MouseGesture():
    def __init__(self) -> None:
        # 마우스 위치 값 임시 저장을 위한 변수
        self.x0, self.y0 = -1,-1

        with open('label_x_07_v2.txt', 'r') as f:
            lines = f.readlines()
            self.start_index = len(lines) + 1
            print(self.start_index)
        self.is_clicked = 0
        self.cnt = 1
        ratio = 0.7
        self.height = int(120 * ratio)
        self.h = 60

        self.img = cv2.imread(f'driving_images_2/img_{self.start_index}.jpg')
        self.img = cv2.resize(self.img,(160, 120))
        cv2.namedWindow('select', flags=cv2.WINDOW_GUI_NORMAL)

        self.img = cv2.line(self.img, (0, self.height), (160, self.height), (0, 0, 255), 1)
        self.img = self.img[self.h:]



        cv2.imshow('select', self.img)

    def on_mouse(self, event, x, y, flags, param):
        global cnt
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.imshow('select', self.img)
            self.x0 = x
            self.y0 = y
            self.is_clicked = 1
            print(f"x: {x}, y: {y}")
            img = self.img.copy()
            img = cv2.circle(img, (x, int(120*0.7)-60), 1, (255, 0, 0), -1)
            cv2.imshow('select', img)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.is_clicked:
                self.is_clicked = 0
                print("next")
                with open('label_x_07_v2.txt', 'a') as f:
                    f.write(f'{self.x0}\n')
                self.cnt += 1
                self.img = cv2.imread(f'driving_images_2/img_{self.start_index + self.cnt}.jpg')
                self.img = cv2.resize(self.img, (160, 120))
                self.img = cv2.line(self.img, (0, self.height), (160, self.height), (0, 0, 255), 1)
                self.img = self.img[self.h:]
                cv2.imshow('select', self.img)

            else:
                print("선택 안됨")


        return

mouse_class = MouseGesture()

cv2.setMouseCallback('select', mouse_class.on_mouse)
cv2.waitKey(0)


