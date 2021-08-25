import tensorflow as tf
print(tf.__version__)

#즉시실행모드???
tf.compat.v1.disable_eager_execution()


# print("hello world")

hello = tf.constant("hello world")
print(hello)
# Tensor("Const:0", shape=(), dtype=string)

# sess = tf.Session()
sess = tf.compat.v1.Session()
print(sess.run(hello))
# b'hello world'