import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

#1. 데이터
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

# 모델 구성
w1 = tf.get_variable('w1', shape=[3, 3, 1, 32])  # get_variabel을 쓰면  shape를 명시해 줘야한다.
                                #[kernel_size,input,output]
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')

print(w1) # (3,3,1,32)
print(L1) # (?, 28, 28, 32)
 

# w2 = tf.Variable(tf.random_normal([3, 3, 1, 32]), dtype = tf.float32)
# w3 = tf.Variable([1], dtype= tf.float32)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(np.min(sess.run(sess.run(w1))))
# print(np.max(sess.run(sess.run(w1))))
# print(np.mean(sess.run(sess.run(w1))))
# print(np.median(sess.run(sess.run(w1))))
# print(sess.run(w1))
# print(w1)