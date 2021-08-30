import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())
print(tf.__version__)
# False
# 2.4.1

# tf.set_random_seed(66)

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
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool(L1, ksize=[1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')

# print(w1) # (3,3,1,32)
# print(L1) # (?, 28, 28, 32)
 
w2 = tf.get_variable('w2', shape=[3,3,32,64])
L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.selu(L2)
L2_maxpool = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L2)
print(L2_maxpool)
# Tensor("Selu:0", shape=(?, 14, 14, 64), dtype=float32)
# Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)

w3 = tf.get_variable('w3', shape=[3,3,64, 128])
L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1,1,1,1], padding='SAME')
L3 = tf.nn.selu(L3)
L3_maxpool = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L3)
print(L3_maxpool)

w4 = tf.get_variable('w4', shape=[3,3, 128, 64])
L4 = tf.nn.conv2d(L3_maxpool, w2, strides=[1,1,1,1], padding='VALID')
L4 = tf.nn.selu(L4)
L4_maxpool = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L4)
print(L4_maxpool)

# Flatten
L_flat = tf.reshape(L4_maxpool, [-1, 2*2*64])
print("플래튼 : ", L_flat)

# L5 DNN
w5 = tf.get_variable("w5", shape=[2*2*64,64])
b5 = tf.Variable(tf.random_normal([64]), name='b1')
L5 = tf.matmul(L_flat, w5) + b5
L5 = tf.nn.selu(L5)
L5 = tf.nn.dropout(L5, keep_prob=0.2)
print(L5)

# L6 DNN
w6 = tf.get_variable("w6", shape=[64,32])
b6 = tf.Variable(tf.random_normal([32]), name='b2')
L6 = tf.matmul(L5, w6) + b6
L6 = tf.nn.selu(L6)
L6 = tf.nn.dropout(L6, keep_prob=0.2)
print(L6)

# L7 Softmax
w7 = tf.get_variable("w7", shape=[32,10])
b7 = tf.Variable(tf.random_normal([10]), name='b3')
L7 = tf.matmul(L6, w7) + b7
hypothesis = tf.nn.softmax(L7)
print(hypothesis)

# 3. 컴파일, 훈련
loss = tf.reduce_mean(-tf.reduce_sum(y* tf.log(hypothesis), axis=1))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_loss = 0
    for i in range(total_batch):
        start = i * batch_size
        end = start + batch_size
        batch_x, batch_y = x_train[start:end], y_train[start:end]

        feed_dict = {x: batch_x, y: batch_y}
        batch_loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
        avg_loss += batch_loss/total_batch
    
    print('epoch : ', '%04d' %(epoch + 1), 'loss : {:.9f}'.format(avg_loss))

print("훈련 끝")

prediction = tf.equal(tf.compat.v1.arg_max(hypothesis,1), tf.compat.v1.arg_max(y,1))
accuracy = tf.reduce_mean(tf.compat.v1.cast(prediction, dtype=tf.float32))
print('Acc :', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))