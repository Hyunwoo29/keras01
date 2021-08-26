import tensorflow as tf
tf.compat.v1.set_random_seed(777)

x = [1, 2, 3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.0], tf.float32)

hypothesis = x * W + b 

# [실습]
sess = tf.compat.v1.Session()   # tf.compat.v1 붙이는 이유는 warning뜨는거 귀찮아서.
sess.run(tf.global_variables_initializer())
aaa = sess.run(hypothesis)
# print("aaa : ", aaa)  aaa :  [1.3       1.6       1.9000001]
sess.close() # 세션을 닫아준다. 

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = hypothesis.eval()     # 변수쩜이발 
print("bbb : ", bbb)
sess.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = hypothesis.eval(session=sess)
print("ccc : ", ccc)
sess.close()
