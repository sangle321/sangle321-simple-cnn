import cv2
import tensorflow as tf
from keras.utils import np_utils

mnist = tf.keras.datasets.mnist

(x_train, y_train) , (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_train /= 255

y_train = np_utils.to_categorical(y_train, 10)

# Tạo model CNN
model = tf.keras.models.Sequential([
    #Loo convol
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), input_shape=(28, 28, 1), activation=tf.nn.relu),
    # lop pooling
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #Lop 2
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(300, activation=tf.nn.relu),
    # Dense Định nghĩa một mạng nơ ron fully connected
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# model.summary()
# # [1000, 28, 28, 3]
# """
#     1: Số lượng các sample
#         + Nếu bạn có 10.000 dữ liệu training thì vt1 là 10K
#     2 , 3: Kích thước của ảnh 28*28
#     4: Ảnh màu 3 chiều
# """
#
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=15)
model.save('model.h5')

