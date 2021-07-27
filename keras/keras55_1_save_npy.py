#넘파이로 세이브 하는법 빠르다.
from sklearn.datasets import load_iris, load_boston, load_diabetes, load_breast_cancer, load_wine
import numpy as np
from tensorflow.keras.datasets import mnist,fashion_mnist,cifar10,cifar100

datasets = load_iris()
datasets1 = load_boston()
datasets2 = load_breast_cancer()
datasets3 = load_diabetes()
datasets4 = load_wine()
(m_x_train, m_y_train) , (m_x_test, m_y_test) = mnist.load_data()
(fm_x_train, fm_y_train) , (fm_x_test, fm_y_test) = fashion_mnist.load_data()
(c10_x_train, c10_y_train) , (c10_x_test, c10_y_test) = cifar10.load_data()
(c100_x_train, c100_y_train) , (c100_x_test, c100_y_test) = cifar100.load_data()

x_data_iris = datasets.data
y_data_iris = datasets.target
x_data_boston = datasets1.data
y_data_boston = datasets1.target
x_data_breast_cancer = datasets2.data
y_data_breast_cancer = datasets2.target
x_data_diabetes = datasets3.data
y_data_diabetes = datasets3.target
x_data_wine = datasets4.data
y_data_wine = datasets4.target


# print(type(x_data_iris), type(y_data_iris))
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>

#보스톤, 캔서, 디아벳까지 npy로 세이브
# np.save('./_save/_npy/K55_x_data_iris.npy', arr=x_data_iris) # x_data 저장방법
# np.save('./_save/_npy/K55_y_data_iris.npy', arr=y_data_iris) # y_data 저장방법

# np.save('./_save/_npy/K55_x_data_boston.npy', arr=x_data_boston) # x_data 저장방법
# np.save('./_save/_npy/K55_y_data_boston.npy', arr=y_data_boston) # y_data 저장방법

# np.save('./_save/_npy/K55_x_data_breast_cancer.npy', arr=x_data_breast_cancer) # x_data 저장방법
# np.save('./_save/_npy/K55_y_data_breast_cancer.npy', arr=y_data_breast_cancer) # y_data 저장방법

# np.save('./_save/_npy/K55_x_data_diabetes.npy', arr=x_data_diabetes) # x_data 저장방법
# np.save('./_save/_npy/K55_y_data_diabetes.npy', arr=y_data_diabetes) # y_data 저장방법

# np.save('./_save/_npy/K55_x_data_wine.npy', arr=x_data_wine) # x_data 저장방법
# np.save('./_save/_npy/K55_y_data_wine.npy', arr=y_data_wine) # y_data 저장방법

np.save('./_save/_npy/k55_x_train_mnist.npy', arr=m_x_train)
np.save('./_save/_npy/k55_y_train_mnist.npy', arr=m_y_train)
np.save('./_save/_npy/k55_x_test_mnist.npy', arr=m_x_test)
np.save('./_save/_npy/k55_y_test_mnist.npy', arr=m_y_test)

np.save('./_save/_npy/k55_x_train_fashionmnist.npy', arr=m_x_train)
np.save('./_save/_npy/k55_y_train_fashionmnist.npy', arr=m_y_train)
np.save('./_save/_npy/k55_x_test_fashionmnist.npy', arr=m_x_test)
np.save('./_save/_npy/k55_y_test_fashionmnist.npy', arr=m_y_test)

np.save('./_save/_npy/k55_x_train_cifar10.npy', arr=m_x_train)
np.save('./_save/_npy/k55_y_train_cifar10.npy', arr=m_y_train)
np.save('./_save/_npy/k55_x_test_cifar10.npy', arr=m_x_test)
np.save('./_save/_npy/k55_y_test_cifar10.npy', arr=m_y_test)

np.save('./_save/_npy/k55_x_train_cifar100.npy', arr=m_x_train)
np.save('./_save/_npy/k55_y_train_cifar100.npy', arr=m_y_train)
np.save('./_save/_npy/k55_x_test_cifar100.npy', arr=m_x_test)
np.save('./_save/_npy/k55_y_test_cifar100.npy', arr=m_y_test)