from keras.models import Sequential
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.src.losses import Huber
from keras.src.optimizers import SGD, Adam, RMSprop
from keras.utils import plot_model
from keras import Model
import cv2
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt

img_path = 'driving_images_2/'
label_path = 'label_x_07_v2.txt'

#-----------------load data and preproccessing-------------------#
# #-------------------save data as pickle file
# # load image
# def load_img_data(path, num):
#     train_image = []
#     print("이미지 로딩 중...")
#     for i in range(num):
#         img = cv2.imread(path+f'img_{i+1}.jpg')
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img = cv2.resize(img, (160, 120))
#         img = img[60:]
#         img = img / 255.0
#         train_image.append(img)
#     print("이미지 로딩 완료!")
#     return np.array(train_image)
#
# train_image = load_img_data(img_path, 7075)
#
# # load label(x coordinate)
# def load_label_data(path):
#     print("라벨 로딩 중...")
#     with open(path, 'r') as f:
#         lines = f.readlines()
#         lines = list(map(lambda s: s.strip(), lines)) # 줄바꿈 삭제
#         lines = list(map(lambda s: int(s), lines)) # 정수로 변환
#         #lines = list(map(lambda s: s/160, lines)) # 노멀라이즈
#         print("라벨 로딩 완료!")
#         return np.array(lines)
#
# train_label = load_label_data(label_path)
#
#
# with open('data.pickle', 'wb') as f:
#     pickle.dump((train_image, train_label), f)

#-------------------------------load pickle data
with open('data.pickle', 'rb') as fr:
    train_image, train_label = pickle.load(fr)



# shuffling dataset
def shuffle_data(x_data, y_data):
    data = list(zip(x_data, y_data))
    random.shuffle(data)
    x_data = [d[0] for d in data]
    y_data = [d[1] for d in data]
    return np.array(x_data), np.array(y_data)



train_image, train_label = shuffle_data(train_image, train_label)

# # data visualization
# from matplotlib.patches import Circle
#
# for img, pos in zip(train_image[50:100], train_label[50:100]):
#     fig, ax = plt.subplots(1)
#     ax.set_aspect('equal')
#     ax.imshow(img)
#     circ = Circle((int(pos*160), int(120*0.7)-60), 5)
#     ax.add_patch(circ)
#     plt.show()

print(len(train_image), len(train_label))



#-------------------input setting----------------------#
image = Input(shape=(60, 160, 1)) # 이미지 사이즈 800 x 600 (1/5 해상도로 줄임) = 160 x 120


#---------------------model1-----------------------#

conv1 = Conv2D(20, (3, 3), padding = 'same', activation='tanh', input_shape=(60, 160, 1))(image)
BN1 = BatchNormalization()(conv1)
pool1 = MaxPooling2D(pool_size=(3, 3))(BN1)
dropout1 = Dropout(0.25)(pool1)
flat1 = Flatten()(dropout1)
dense1 = Dense(200, activation='tanh')(flat1)
dropout2 = Dropout(0.5)(dense1)
dense2 = Dense(1, activation='relu')(dropout2)

model = Model(inputs=image, outputs=dense2)


#-------------------check model-----------------------#
plot_model(model, to_file='model.png', show_shapes=True, rankdir='TB')
model.summary()
#--------------------compiling model----------------------#
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
#adam = Adam(lr=0.0005, decay=1e-6, amsgrad=True)
#rmsprop = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
huber = Huber(delta=0.5, reduction="auto", name="huber_loss")

# model.compile(loss='logcosh', optimizer=sgd)
model.compile(loss=huber, optimizer=sgd)
n = 200 # 확인용 데이터 나누기 갯수
model.fit(train_image[:-n], train_label[:-n], validation_data=(train_image[-n:], train_label[-n:]), batch_size=8, epochs=70, verbose = 1)

#model.evaluate(train_image[-60:], train_label[-60:])

model.save('middle_point.h5')