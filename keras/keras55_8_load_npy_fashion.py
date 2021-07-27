import numpy as np

x_train =np.load('./_save/_npy/k55_x_train_fashionmnist.npy')
y_train = np.load('./_save/_npy/k55_y_train_fashionmnist.npy')
x_test = np.load('./_save/_npy/k55_x_test_fashionmnist.npy')
y_test = np.load('./_save/_npy/k55_y_test_fashionmnist.npy')

print(x_train)
print(y_train)
print(x_test)
print(y_test)
