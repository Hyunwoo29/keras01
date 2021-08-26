# 실습
from sklearn.datasets import load_boston
import tensorflow as tf

tf.set_random_seed(66)

datasets = load_boston()

x = datasets.data 
y = datasets.target
print(x.shape, y.shape) 