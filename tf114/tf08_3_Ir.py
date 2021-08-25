# 실습
# tf08_2 파일의 Ir을 수정해서
# epcoh가 2000번이 아니라 100번 이하로 줄여라
# 결과치는 step=100 이하, w=1.9999, b=0.9999

# 실습
# 1. [4]
# 2. [5,6]
# 3. [6,7,8]

# y = wx + b
import tensorflow as tf
tf.set_random_seed(66)

# x_train = [1, 2, 3]  #  w= 1, b = 0
# y_train = [3, 5, 7]

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None]) 

x_test = tf.placeholder(tf.float32, shape=[None])
y_test = tf.placeholder(tf.float32, shape=[None]) 

# W = tf.Variable(1, dtype=tf.float32, name='test')
# b = tf.Variable(1, dtype = tf.float32)

W = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)


hypothesis = x_test * W + b # 모델구현
# f(x) = wx + b

loss = tf.reduce_mean(tf.square(hypothesis - y_test)) # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.04)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


with tf.Session() as sess : #with문을쓰면 자동으로 close역할도 수행한다.
    sess.run(tf.compat.v1.global_variables_initializer()) #변수 초기화
    for step in range(101):
        # sess.run(train)
        _, loss_val, W_val, b_val = sess.run([train, loss, W, b],
                feed_dict={x_test:[1,2,3], y_test:[1,2,3]})
        if step % 1 ==0:
            # print(step, sess.run(loss), sess.run(W), sess.run(b))
            print(step, loss_val, W_val, b_val)
    
    print('[4] 예측결과 : ', sess.run(hypothesis, feed_dict = {x_test:[4]}))
    print('[5, 6] 예측결과 : ', sess.run(hypothesis, feed_dict = {x_test:[5,6]}))
    print('[6, 7, 8] 예측결과 : ', sess.run(hypothesis, feed_dict = {x_test:[6,7,8]}))

print('끗')


# predict 하는 코드를 추가하시오!
# x_test라는 placholder  생성!
