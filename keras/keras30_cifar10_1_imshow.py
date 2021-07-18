import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
# x_train.shape: (50000, 32, 32, 3), y_train.shape: (50000,1)
# x_test.shape: (10000, 32, 32, 3), y_test.shape: (10000, 1)

# print(x_train[0])
# print(y_train[0])

plt.imshow(x_train[100], 'BuPu')
plt.show()
#완성하시오!
# 이미지가 32, 32, 3