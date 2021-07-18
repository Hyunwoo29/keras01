import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

#(60000, 28, 28) (60000,)
# (10000, 28, 28) (10000,)

# print(x_train[0])
# print("y[0] 값 : ", y_train[0])


# plt.imshow(x_train[0], 'gray')
# plt.show()

print(x_train[111])
print("y[111] 값 : ", y_train[111])


plt.imshow(x_train[111], 'gray')
plt.show()

#그레이 색상으로 이미지 출력
