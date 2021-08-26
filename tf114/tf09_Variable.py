import tensorflow as tf
tf.compat.v1.set_random_seed(777)

W = tf.Variable(tf.random_normal([1]), name='weight')
print(W)
# <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref> # 자료형

sess = tf.compat.v1.Session()   # tf.compat.v1 붙이는 이유는 warning뜨는거 귀찮아서.
sess.run(tf.global_variables_initializer())
aaa = sess.run(W)
# print("aaa : ", aaa) # aaa :  [2.2086694]
sess.close() # 세션을 닫아준다. 

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = W.eval()     # 변수쩜이발 
print("bbb : ", bbb)
sess.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = W.eval(session=sess)
print("ccc : ", ccc)
sess.close()