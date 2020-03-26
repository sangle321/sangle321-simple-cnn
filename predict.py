import cv2
from keras.initializers import glorot_uniform
from tensorflow.keras.models import load_model
from keras.utils import CustomObjectScope

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model('model.h5')

img_test = cv2.imread('img/anh4.png',0)

img_test = img_test.reshape(1, 28, 28, 1)

print(model.predict(img_test))