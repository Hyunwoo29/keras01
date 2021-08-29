# Dnn
# sigmoid, linear
# 단층 퍼셉트론으로 구성

import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

datasets = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = datasets.load_data()
# print(x_train.shape, y_test.shape) (60000, 28, 28) (10000,)

y_train = y_train.reshape(-1,1)
# print(y_train.shape) (60000, 1)
y_test = y_test.reshape(-1,1)
# print(y_test.shape) (10000, 1)

x_train = x_train.reshape(-1, 28*28)/255.
x_test = x_test.reshape(-1, 28*28)/ 255.

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train).toarray()
y_test = encoder.transform(y_test).toarray()
# print(y_train.shape) (60000, 10)

x = tf.placeholder(tf.float32, shape = [None, 28*28])
y = tf.placeholder(tf.float32, shape = [None, 10])

w = tf.compat.v1.Variable(tf.zeros([28*28, 10]), name='weight')
b = tf.compat.v1.Variable(tf.zeros([1,10]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

cost = tf.reduce_mean(-tf.reduce_sum(y* tf.log(hypothesis), axis=1)) # categorical_crossentropy
optimizer = tf.train.AdamOptimizer(learning_rate=0.017).minimize(cost)

from sklearn.metrics import accuracy_score

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(201) :
        cost_val, _ = sess.run([cost, optimizer],
            feed_dict={x:x_train, y: y_train})
        if step % 10 == 0:
            print(step, "cost : ", cost_val)

    predict = sess.run(hypothesis, feed_dict={x:x_test})
    print(sess.run(tf.argmax(predict, 1)))

    y_pred = sess.run(hypothesis, feed_dict = {x:x_test})
    y_pred = np.argmax(y_pred, axis= 1)
    y_test = np.argmax(y_test, axis= 1)
    print('acc_score : ', accuracy_score(y_test, y_pred))

# 결과값
# [7 2 1 ... 4 5 6]
# acc_score :  0.9269