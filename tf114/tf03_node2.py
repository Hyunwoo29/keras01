# 실습
# 덧셈
# 뺄셈
# 곱셈
# 나눗셈

import tensorflow as tf
from tensorflow.python.client.session import Session
from tensorflow.python.ops.math_ops import subtract
node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

node3 = tf.add(node1, node2)  # 덧셈
node4 = tf.subtract(node1, node2) # 뺄셈
node5 = tf.multiply(node1, node2) # 곱셈
node6 = tf.divide(node1, node2)  # 나눗셈

sess = Session()
print('더하기 : ', sess.run(node3))
print('빼기 : ', sess.run(node4))
print('곱하기 : ', sess.run(node5))
print('나누기 : ', sess.run(node6))




