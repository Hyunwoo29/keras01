import tensorflow as tf
sess = tf.Session()

x = tf.Variable([2], dtype=tf.float32, name='test')

# 변수를 사용하고 싶으면 반드시 global_variables_initializer을 써야한다.
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(x))