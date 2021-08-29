# 실습
from sklearn.datasets import load_boston
import tensorflow as tf

tf.set_random_seed(66)

datasets = load_boston()

x = datasets.data 
y = datasets.target
# print(x.shape, y.shape) (506, 13) (506,)
from sklearn.model_selection import train_test_split

y = y.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2,  random_state = 66)

x = tf.placeholder(tf.float32, shape = [None, 10])
y = tf.placeholder(tf.float32, shape = [None, 1])

# 최종 결론값은 r2_score로 할것

w = tf.Variable(tf.random_normal([10,1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = tf.matmul(x,w) + b

cost = tf.reduce_mean(tf.square(hypothesis -y))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.8) # larning_rate를 줄여봤다.
train = optimizer.minimize(cost)

from sklearn.metrics import r2_score
with tf.Session() as sess : #with문을쓰면 자동으로 close역할도 수행한다.
    sess.run(tf.compat.v1.global_variables_initializer()) #변수 초기화
    for step in range(5001) :
        cost_val, hy_val, _ = sess.run([cost,hypothesis,train], 
                            feed_dict={x:x_train, y:y_train})       # 트레인 데이터로 훈련시킨다.
        if step % 1000 == 0 :
            print(step, "cost : ", cost_val)

    predict = sess.run(hypothesis, feed_dict = {x:x_test})          # 훈련 데이터로 뽑은 가중치로 test 데이터에 대한 결과를 예측한다.
    r2 = r2_score(y_test,predict)                                   # 테스트 값을 예측하여 얻은 예측값과 y_test값을 비교한다.
    print("R2 : ",r2)