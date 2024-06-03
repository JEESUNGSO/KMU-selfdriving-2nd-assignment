#! /usr/bin/env python3

#-----------------필요한 모듈 및 토픽 타입 불러오기--------------------
import rospy # ros를 파이썬에서 사용하기 위한 라이브러리

#---토픽 타입 불러오기
from xycar_msgs.msg import xycar_motor # 차량 컨트롤 토픽 타입 (publish)
from sensor_msgs.msg import Image # 카메라 센서 토픽 타입 (subscribe)
from sensor_msgs.msg import Imu # imu 센서 토픽 타입 (subscribe)
from xycar_msgs.msg import xycar_motor # 자량 컨트롤 토픽 타입 (subscribe)

#---추가 라이브러리
from cv_bridge import CvBridge, CvBridgeError # 카메라로 불러온 값을 opencv에서 사용할 수 있도록 하는 라이브러리
import cv2 # opencv
import numpy as np # 넘파이...
from time import sleep
from keras.models import load_model


#---사용자 라이브러리
from get_lane import *
from Birdseye_view import *
from sliding_window import *
from PID_contorl import *
from checkTrafficLight import *

#-----------------콜백 함수 모음--------------------------------------
cam_image = np.empty(shape=[0]) # 이미지 초기화
def img_callback(data):
    #---카메라 콜벡 함수
    global cam_image
    cam_image = CvBridge() .imgmsg_to_cv2(data, "bgr8")


#imu_values =
def imu_callback(data):
    #---imu 센서 콜백함수
    global imu_values
    pass

#-----------------구독, 발행 정하기--------------------------------------
rospy.init_node('control_node', anonymous=True) # 노트 초기화

#---subscirber 설정
rospy.Subscriber('usb_cam/image_raw', Image, img_callback) # 카메라 토픽 콜백함수와 연결
#rospy.Subscriber('carla_ctl/imu', Imu, imu_callback) # imu 토픽 콜백함수와 연결

#---publisher 설정
rate = 0.1 # publish 하는 주기 (단위: 초)
ctl_pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1) # 컨트롤 값 publisher 설정하기


#-----------------필요한 함수들 정의----------------------------------------
var = xycar_motor() # 토픽 타입 변수 생성
def drive(angle, speed):
    #---차량 컨트롤 함수
    # 두개 다 float 타입
    var.angle = angle
    var.speed = speed
    ctl_pub.publish(var)


def get_lane_mask(image):
    height, width = image.shape[:2] # 800 600
    vertices = np.array([
        [0, height * 0.9],
        [width, height * 0.9],
        [width * 0.8, height * 0.6],
        [width * 0.2, height * 0.6]
    ], np.int32)

    return get_lane(image, vertices)



def is_similar(a, b):
    #---a,b가 유사한지 확인하는 함수
    epsilon = 0.1 # 유사한 정도 허용 오차
    diff = abs(a - b)
    return diff < epsilon

middle_points = [-1, -1, -1] # 3개의 평균을 이용해서 흔들림을 최소화
def moving_average_filter(x):
    global middle_points
    # 처음 에는 3개의 값이 없으므로 초기값을 3개의 값으로 사용
    if middle_points[0] == -1:
        middle_points = [x, x, x]
    mean = np.mean(middle_points) # 평균 구함
    # 값 업데이트
    middle_points[0] = middle_points[1]
    middle_points[1] = middle_points[2]
    middle_points[2] = x
    return int(mean)

def get_middle_point(image, model):
    image_gray = image.copy()
    image_gray = cv2.resize(image_gray, (160, 120))
    image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)
    image_gray = image_gray[60:]
    image_norm = image_gray / 255.0
    image_norm = np.expand_dims(image_norm, axis=0)
    x_pred = model.predict(image_norm, verbose=0)[0][0]
    filtered_x_pred = moving_average_filter(x_pred) # 이동 평균 필터 적용
    #if abs(filtered_x_pred - x_pred) > 5:
    #    print(f"x_pred: {filtered_x_pred}, filtered_x_pred: {filtered_x_pred}")
    return filtered_x_pred * 5, int(120*0.7) * 5



#-----------------데이터 처리 및 실행--------------------------------------
model = load_model('/home/sungso/catkin_ws/src/assignment_2/src/middle_point.h5')

# 신호등 roi
x = 727
y = 75
w = 60
h = 60

speed = 0 # 시작시 정지

#cnt = 1 # 이미지 인덱스
f_time = rospy.get_time() # publish를 주기마다 해주기 위해
while not rospy.is_shutdown():
    # 시작 신호 기다리기
    if speed == 0 and check_traffic_light(cam_image, x, y, w, h, 180):
        speed = 0.55 # 속도 설정


    #---차선검출하기
    # lane_mask = get_lane_mask(cam_image)
    # cv2.imshow('lane_mask', lane_mask)


    #---슬라이딩 윈도우 적용하기
    # # 이미지 birds eye view로 변환
    # mask_img = bird_eye_view(mask_img, 0, int(mask_img.shape[0] / 1.5), 444, 120)
    # # 슬라이딩 윈도우
    # result = get_windows(mask_img, 20, 50, 0.3)
    # # 출력
    # cv2.imshow('mask_img', mask_img)
    # cv2.waitKey(0)


    #---차량 전진시키기
    if is_similar(rospy.get_time() - f_time, rate):  # 주기가 되었을때만
        # ---이미지 출력하기
        cv2.imshow('camera', cam_image)


        #---딥러닝 이용한 차선 중앙 검출
        pt_x, pt_y = get_middle_point(cam_image, model)
        cv2.circle(cam_image, (pt_x, pt_y), 10, (255, 0, 0), -1)

        #---PID 제어값 계산
        u = get_u(400, pt_x) # 화면 중앙 값: 400, 중앙 위치값: pt_x
        #print(400 - pt_x)
        drive(u, speed)

        # 이미지 저장
        # if cam_image.shape[0] != 0:
        #     rospy.loginfo(cv2.imwrite(f'/home/sungso/catkin_ws/src/assignment_2/src/driving_images_2/img_{1600+cnt}.jpg', cam_image))
        #     cnt += 1
        f_time = rospy.get_time()  # 시작시간 업데이트

    cv2.waitKey(1)

