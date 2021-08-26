import tensorflow as tf
import numpy as np

tf.set_random_seed(66)

x_data = [[73,51,65],[92,98,11],[89,31,33],[99,33,100],[17,66,79]] # 5행3열
y_data = [[152],[185],[180],[205],[142]]    # 5행 1열

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x, w) + b

# 만약 x가 (5,3)이면 w 도 (3,4) x의 열과 w의 행을 맞춰줘야한다.
# bias가 1이면 결과값은 x의 열을 뺀것과 w의 행을 빼고 합친 (5,4)의 행렬이 나와야한다.

cost = tf.reduce_mean(tf.square(hypothesis -y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.00001) # 1e-5---> 0.00001 # larning_rate를 줄여봤다.
# 1 + 1e-5 -->  1.00001
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(2001):
    cost_val, hy_val, _ = sess.run([cost,hypothesis,train], 
                        feed_dict={x:x_data, y:y_data})
    if epochs % 10 == 0 :
        print(epochs, "cost : ", cost_val, "\n",hy_val)