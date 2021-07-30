from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.python.keras.backend import zeros
(x_train, y_train), (x_test,y_test) = fashion_mnist.load_data()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255, # 0~ 255 사이니까 0~1로 만들어주기위해 255로 나눠줌
    horizontal_flip=True, # 수평이동
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.5,
    fill_mode='nearest' # nearest: 근접   근접일시 어느정도 매칭을 시켜라
)


# 1. ImageDataGenerator를 정의
# 2. 파일에서 땡겨올려면 -> flow_from_directory() / x,y가 튜플형태로 뭉쳐있음
# 3. 데이터에서 땡겨올려면 -> flow()  / x,y 가 나눠있음

# augument_size=100
# x_data = train_datagen.flow(
#             np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1), # x값
#             np.zeros(augument_size), # y --> 임의로 augument_size 100개 집어넣은것. 사이즈 맞추기위해
#             batch_size=augument_size,
#             shuffle=False
# )    # iterator 방식으로 반환!

# # print(type(x_data)) # <class 'tensorflow.python.keras.preprocessing.image.NumpyArrayIterator'>
# # print(type(x_data[0])) # <class 'tuple'>
# # print(type(x_data[0][0]))  # <class 'numpy.ndarray'>
# # 넘파이 타입에서만 shape 찍을수 있다.
# print(x_data[0][0].shape) # (100, 28, 28, 1)  --> x 값
# print(x_data[0][1].shape) # (100,)   --> y 값

# import matplotlib.pyplot as plt
# plt.figure(figsize=(7,7))
# for i in range(49):
#     plt.subplot(7, 7, i)
#     plt.axis('off')
#     plt.imshow(x_data[0][i], cmap='gray')

# plt.show()

# iterator 방식! 뒤에 .next() 넣어주면 됨.
augument_size=100
x_data = train_datagen.flow(
            np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1), # x값
            np.zeros(augument_size), # y --> 임의로 augument_size 100개 집어넣은것. 사이즈 맞추기위해
            batch_size=augument_size,
            shuffle=False
).next()   # iterator 방식으로 반환!

# print(type(x_data)) # <class 'tuple'>
# print(type(x_data[0])) # <class 'numpy.ndarray'>
# 넘파이 타입에서만 shape 찍을수 있다.
# print(x_data[0].shape) # (100, 28, 28, 1)  --> x 값
# print(x_data[1].shape) # (100,)   --> y 값

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')

plt.show()