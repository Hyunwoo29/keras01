import tensorflow as tf
tf.set_random_seed(66)

# 1.데이터
x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]] # (6, 2)
y_data = [[0],[0],[0],[1],[1],[1]]       # (6, 1)

# 2. 모델구성
x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

# 만약 x가 (5,3)이면 w 도 (3,4) x의 열과 w의 행을 맞춰줘야한다.
# bias가 1이면 결과값은 x의 열을 뺀것과 w의 행을 빼고 합친 (5,4)의 행렬이 나와야한다.

# cost = tf.reduce_mean(tf.square(hypothesis -y))
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy 구현

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01) # 1e-5---> 0.00001 # larning_rate를 줄여봤다.
# 1 + 1e-5 -->  1.00001
train = optimizer.minimize(cost)


sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

# 훈련
for epochs in range(5001):
    cost_val, hy_val, _ = sess.run([cost,hypothesis,train], 
                        feed_dict={x:x_data, y:y_data})
    if epochs % 200 == 0 :
        print(epochs, "cost : ", cost_val, "\n",hy_val)

# 평가,예측
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

c,a = sess.run([predicted, accuracy], feed_dict={x: x_data, y: y_data})
print("Hypothesis(예측값) : \n", hy_val, "\n predict(원래값) : \n" ,c , "\n Accuarcy : ",a)

sess.close()