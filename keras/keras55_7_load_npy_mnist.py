import numpy as np

x_train =np.load('./_save/_npy/k55_x_train_mnist.npy')
y_train = np.load('./_save/_npy/k55_y_train_mnist.npy')
x_test = np.load('./_save/_npy/k55_x_test_mnist.npy')
y_test = np.load('./_save/_npy/k55_y_test_mnist.npy')

print(x_train)
print(y_train)
print(x_test)
print(y_test)