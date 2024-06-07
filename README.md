# 국민대학교 자율주행 경진대회 2차 예선과제
## 1. 전체적인 흐름
  1. 신호등 인식
     1. 신호등 초록색 부분 ROI
     2. Green채널의 임계값 이상의 값들의 개수가 특정 값 이상이면 출발
  3. 이 후에는 아래와 같은 흐름으로 동작함
    
  ![Untitled-2024-06-07-2032(1)](https://github.com/JEESUNGSO/KMU-selfdriving-2nd-assignment/assets/166119462/5711b53d-69b4-43b9-a742-95ce8400fb12)

## 2. 색상을 이용한 차선검출이나 canny edge를 이용하지 않은 이유
  1. 너무 밝은 햋빛
     - 너무 밝은 빛이 차선을 없어지게 하여 검출에 어려움이 있었음
       
     ![light](https://github.com/JEESUNGSO/KMU-selfdriving-2nd-assignment/assets/166119462/0d382714-02af-4fb2-af4e-57159dc0cd07)
  2. 그림자
     - 그림자로 인해 색상 차이가 심해져 검출에 어려움이 있었음
       
     ![shadow](https://github.com/JEESUNGSO/KMU-selfdriving-2nd-assignment/assets/166119462/747241b0-9856-47f3-bcfd-f744bba7153a)
  3. 그림자와 빛의 조합
     - 이 둘이 만나면 검출하기 매우 애매모호한 상황이 생김
       
     ![shadow_and_light](https://github.com/JEESUNGSO/KMU-selfdriving-2nd-assignment/assets/166119462/0ca97d8c-e862-4fb5-a5d1-bd82612a3cc6)

## 3. 딥러닝 모델 제작 과정
  1. 직접 다양한 주행을 통해 7000장의 이미지를 확보
  2. labeling.py를 이용해 일정 높이에서의 차선 중앙 좌표를 라벨링
  3. CNN 모델 구조
     - 처음에 accuracy를 넣어서 모델을 훈련했는데 수렴하지 않았음 --> accuracy는 분류 모델에서 사용하는 것, 연속적인 값을 예측하므로 회귀 모델을 만들어야했음
       
     ![model](https://github.com/JEESUNGSO/KMU-selfdriving-2nd-assignment/blob/main/data_processing/model.png?raw=true)
  4. training
     - 약간 과적합 된 경향이 있었음. 그러나 주행해야하는 환경이 변하지 않고 빠르게 달리는게 주 목적이고, 제출 프로그램에 문제가 발생하여 해결할 시간을 확보하고자 개선없이 마무리 하였음
     - 
## 4. PID 컨트롤
  - [1차 과제](https://github.com/JEESUNGSO/KMU-selfdriving-1st-assignment)를 참고하기를 바람
  - 
## 5. 결과
  - 이미지를 클릭하시면 유튜브 영상으로 연결 됩니다.
    
  [![주행 영상](https://i9.ytimg.com/vi/GvpFaJiU7H0/mqdefault.jpg?sqp=CPiDjLMG&rs=AOn4CLAfZ22uk9EJoIOzajsil0EY6fFm-Q&retry=4)](https://youtu.be/GvpFaJiU7H0)
  
## 6. 개선할 점
  1. 가장큰 문제는 직선구간 가속후 회전시 e값의 변화가 점점 커지기 때문에 속도에 비해 u(steering)값이 느리게 변해 벽과 중돌이 발생.
     - VISUAL SLAM을 이용하여 만들면 차량의 위치를 알 수 있기에 해결이 가능함
     - Radar SLAM을 이용하기를 원했지만 시뮬레이터의 문제로 가드레일이 레이저 흡수율이 100%로 radar 센서가 작동하지않아 사용 불가했음
  3. 부적합한 모델사용, 데이터 부족, 과적합 등으로 인해서 정확학 예측 불가
     - 딥러닝 레이어의 대략적인 역할은 알지만 어떻게 모델을 최적화 하는지, 데이터가 부족한 것인지, 과적합 된 것인지를 알 수가 없었고, 시간상 여유가 부족하여 추가적인 자료조사를 하지 못했음 본대회에서 이러한 모델을 사용하게 된다면 가장 먼저 해결해야할 문제라고 생각됨
