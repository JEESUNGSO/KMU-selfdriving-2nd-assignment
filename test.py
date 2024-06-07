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
from keras.models import load_model


#---사용자 라이브러리
from PID_contorl import *
from checkTrafficLight import *
from Birdseye_view import *
from get_lane import *

#-----------------콜백 함수 모음--------------------------------------
cam_image = np.empty(shape=[0]) # 이미지 초기화

last_callback = 0.0
def img_callback(data):
    global last_callback
    #---callback period
    gap = rospy.get_time() - last_callback
    if gap > 0.5:
        rospy.loginfo(f"time gap was {gap}s")
    last_callback = rospy.get_time()

    #---카메라 콜벡 함수
    global cam_image
    cam_image = CvBridge() .imgmsg_to_cv2(data, "bgr8")


    


#-----------------구독, 발행 정하기--------------------------------------
rospy.init_node('control_node', anonymous=True) # 노트 초기화

#---subscirber 설정
rospy.Subscriber('usb_cam/image_raw', Image, img_callback) # 카메라 토픽 콜백함수와 연결

#---publisher 설정
ctl_pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1) # 컨트롤 값 publisher 설정하기


#-----------------필요한 함수들 정의----------------------------------------
var = xycar_motor() # 토픽 타입 변수 생성
def drive(angle, speed):
    #---차량 컨트롤 함수
    # 두개 다 float 타입
    var.angle = angle
    var.speed = speed
    ctl_pub.publish(var)


def is_similar(a, b):
    #---a,b가 유사한지 확인하는 함수
    epsilon = 0.3 # 유사한 정도 허용 오차
    diff = abs(a - b)
    return diff < epsilon


def get_middle_point(image, model):
    image_gray = image.copy()
    image_gray = cv2.resize(image_gray, (160, 120))
    image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)
    image_gray = image_gray[60:]
    image_norm = image_gray / 255.0
    image_norm = np.expand_dims(image_norm, axis=0)
    x_pred = int(model.predict(image_norm, verbose=0)[0][0])
    return x_pred * 5, int(120*0.7) * 5

def speed_multiplier(x, mul, min_, is_starting):
    if is_starting:
        return 1.0
    else:
        x *= mul
        return max([min_, -abs(x) + 1])


def u_multiplier(speed, mul, min_, diff, threshold):
    if diff > threshold:
        #like corner
        #rospy.loginfo(f"!!!!!!!!!!!!!!!!!!!!!huge action!!!!!!!!!!!!!!!!!!")

        return 1.4
    else:
        x = speed * mul
        return max([min_, -abs(x) + 1])


def rapid_speed_control(u, speed, breaking):
    if breaking:
        #print("lkllkllklklklklklklklklkl(breaking)lklklklklklklklklklklklklklklk")
        drive(u, 0)
    else:
        drive(u_w, speed)


#-----------------데이터 처리 및 실행--------------------------------------
try:
    model = load_model('/home/sungso/catkin_ws/src/assignment_2/src/middle_point.h5')
except:
    model = load_model('/home/divinetech/catkin_ws/src/assignment_2/src/middle_point.h5')
speed = 0 # 시작시 정지 (토픽에 전달하는 speed는 아마 최대 속력에 곱해지는 값인것 같음)

# 신호등 roi
x = 727
y = 75
w = 60
h = 60

SPEED = 0.85 # 속도 최댓값
is_driving = False

is_starting = True

rate = 0.1 # publish 하는 주기 (단위: 초)

cnt = 0 # repeat count

f_time = rospy.get_time() # publish를 주기마다 해주기 위해
while not rospy.is_shutdown():
    
    # 시작 신호 기다리기
    if not is_driving and check_traffic_light(cam_image, x, y, w, h, 100):
        rospy.loginfo("-----start--------------------------------!!!!!!")
        is_driving = True

    if cnt == 100:
        is_starting = False
        rospy.loginfo("initial speed is done")
        cnt = 999
    elif cnt < 100:
        cnt += 1

    #---차량 전진시키기
    if is_similar(rospy.get_time() - f_time, rate):  # 주기가 되었을때만
        f_time = rospy.get_time()  # 시작시간 업데이트

        if is_driving and len(cam_image.shape) != 1:
            #---딥러닝 이용한 차선 중앙 검출
            pt_x, pt_y = get_middle_point(cam_image, model)
            cv2.circle(cam_image, (pt_x, pt_y), 10, (255, 0, 0), -1)

            #---PID 제어값 계산
            u = get_u(400, pt_x) # 화면 중앙 값: 400, 중앙 위치값: pt_x
            
            speed = SPEED * speed_multiplier(u, 5.5, 0.85, is_starting)
            threshold = 100
            diff = abs(400 - pt_x)
            u_w = u * u_multiplier(speed, 0.75, 0.6, diff, threshold)
            

            rospy.loginfo(f"speed: {speed:0.4f},\t u: {u:0.4f},\t u_w: {u_w:0.4f},\t diff: {400 - pt_x}")

            #---회전값에 의해 차량 속도 제어
            rapid_speed_control(u_w, speed, False)

            #---camera
            cv2.imshow("cam", cam_image)
            cv2.waitKey(1)

        


