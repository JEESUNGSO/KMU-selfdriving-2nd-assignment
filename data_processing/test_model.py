from keras.models import load_model
import numpy as np
import cv2

img = cv2.imread('driving_images/img_642.jpg')
img_gray = img.copy()
img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

img_norm = img_gray / 255.0

img_norm = np.expand_dims(img_norm, axis=0)
print(img_norm.shape)

model = load_model('middle_point.h5')

x_pred = model.predict(img_norm)[0][0] * 160


img = cv2.circle(img, (int(x_pred), int(120*0.7)), 2, (0, 0, 255), -1)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()