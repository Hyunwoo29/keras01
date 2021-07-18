from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
#  x_train.shape: (60000, 28, 28), y_train.shape: (60000,)
#  x_test.shape: (10000, 28, 28), y_test.shape: (10000,)

# print(x_train[0])
# print(y_train[0])

plt.imshow(x_train[100], 'gray')
plt.show()

#완성하시오!
