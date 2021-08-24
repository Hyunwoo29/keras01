import tensorflow as tf
print(tf.__version__)

# print("hello world")

hello = tf.constant("hello world")
print(hello)
# Tensor("Const:0", shape=(), dtype=string)

# sess = tf.Session()
sess = tf.compat.v1.Session()
print(sess.run(hello))
# b'hello world'