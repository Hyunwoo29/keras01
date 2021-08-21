import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. Data

x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. Model

model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(2))
model.add(Dense(1))

# print(model.weights)
print("=============================")
print(model.trainable_weights)
# model.summary()

# numpy=array([[-0.9228578 , -0.06822741,  1.1248454 ]], # 첫번째 w값은 [] 초기화 되있다 라는뜻.  dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
# array([[ 0.5514213 , -0.59041625],
#        [ 0.6510539 ,  0.43706036],
#        [ 0.01306677,  0.7080344 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
# array([[-0.92237836],
#        [-1.1563915 ]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]