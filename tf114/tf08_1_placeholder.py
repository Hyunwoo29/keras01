# y = wx + b
import tensorflow as tf
tf.set_random_seed(66)

# x_train = [1, 2, 3]  #  w= 1, b = 0
# y_train = [3, 5, 7]

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None]) 

# W = tf.Variable(1, dtype=tf.float32, name='test')
# b = tf.Variable(1, dtype = tf.float32)

W = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)


hypothesis = x_train * W + b # 모델구현
# f(x) = wx + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train)) # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    # sess.run(train)
    _, loss_val, W_val, b_val = sess.run([train, loss, W, b],
            feed_dict={x_train:[1,2,3], y_train:[1,2,3]})
    if step % 20 ==0:
        # print(step, sess.run(loss), sess.run(W), sess.run(b))
        print(step, loss_val, W_val, b_val)


print('끗')
