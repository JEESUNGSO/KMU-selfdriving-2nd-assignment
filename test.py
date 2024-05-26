#! /usr/bin/env python3

#-----------------필요한 모듈 및 토픽 타입 불러오기--------------------
import rospy # ros를 파이썬에서 사용하기 위한 라이브러리

#---토픽 타입 불러오기
from xycar_msgs.msg import xycar_motor # 차량 컨트롤 토픽 타입 (publish)
from sensor_msgs.msg import Image # 카메라 센서 토픽 타입 (subscribe)
from sensor_msgs.msg import LaserScan # 라이다 센서 토픽 타입 (subscribe)
from sensor_msgs.msg import Imu # imu 센서 토픽 타입 (subscribe)
from xycar_msgs.msg import xycar_motor # 자량 컨트롤 토픽 타입 (subscribe)

#---추가 라이브러리
from cv_bridge import CvBridge, CvBridgeError # 카메라로 불러온 값을 opencv에서 사용할 수 있도록 하는 라이브러리
import cv2 # opencv
import numpy as np # 넘파이...
from time import sleep

#-----------------콜백 함수 모음--------------------------------------
cam_image = np.empty(shape=[0]) # 이미지 초기화
def img_callback(data):
    #---카메라 콜벡 함수
    global cam_image
    cam_image = CvBridge() .imgmsg_to_cv2(data, "bgr8")

lidar_values = np.empty(shape=[0]) # 라이다 센서값 초기와
def lidar_callback(data):
    #---라이다 센서 콜백함수
    global lidar_values
    lidar_values = data

#imu_values =
def imu_callback(data):
    #---imu 센서 콜백함수
    global imu_values
    pass

#-----------------구독, 발행 정하기--------------------------------------
rospy.init_node('control_node', anonymous=True) # 노트 초기화

#---subscirber 설정
rospy.Subscriber('usb_cam/image_raw', Image, img_callback) # 카메라 토픽 콜백함수와 연결
rospy.Subscriber('carla_ctl/scan', LaserScan, lidar_callback) # 라이다 토픽 콜백함수와 연결
rospy.Subscriber('carla_ctl/imu', Imu, imu_callback) # imu 토픽 콜백함수와 연결

#---publisher 설정
rate = 1 # publish 하는 주기 (단위: 초)
ctl_pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1) # 컨트롤 값 publisher 설정하기


#-----------------필요한 함수들 정의----------------------------------------
var = xycar_motor() # 토픽 타입 변수 생성
def drive(angle, speed):
    #---차량 컨트롤 함수
    # 두개 다 float 타입
    var.angle = angle
    var.speed = speed
    ctl_pub.publish(var)

def print_img(image):
    #---이미지 출력함수
    if image.shape[0] == 0: # 이미지가 정상적으로 불러와질 때만
        #rospy.loginfo("Image load failed (It's happened at beginning)")
        return
    cv2.imshow('camera', image)
    cv2.waitKey(1)


def is_similar(a, b):
    #---a,b가 유사한지 확인하는 함수
    epsilon = 0.1 # 유사한 정도 허용 오차
    diff = abs(a - b)
    return diff < epsilon

#-----------------데이터 처리 및 실행--------------------------------------
sleep(5) # 5초뒤 출발 (디버깅용)


f_time = rospy.get_time() # publish를 주기마다 해주기 위해
while not rospy.is_shutdown():
    # ---라이다 센서 출력하기
    rospy.loginfo(f"Lidar values: {lidar_values}")

    #---이미지 출력하기
    print_img(cam_image)


    #---imu 센서 출력하기

    #---차량 전진시키기

    if is_similar(rospy.get_time() - f_time, rate): # 주기가 되었을때만
        #rospy.loginfo("forward!!!")
        drive(0, 0.3)
        f_time = rospy.get_time() # 시작시간 업데이트

