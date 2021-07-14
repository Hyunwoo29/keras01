import numpy as np
#1. 데이터
x = np.array([range(100), range(301, 401), range(1, 101),
                range(100), range(401, 501)])

print(x.shape) # (5, 100)
x = np.transpose(x)
print(x.shape) # (100, 5)

y = np.array([range(711, 811), range(101, 201)])
y = np.transpose(y)
print(y.shape) # (100, 2)

#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 함수형 모델구성 
input1 = Input(shape=(5,))
dense1 = Dense(3)(input1)
dense2 = Dense(4)(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(2)(dense3)

model = Model(inputs=input1, outputs=output1)

model.summary()

# 위에꺼랑 밑에꺼 딱 두가지 모델이 있다.
# 이유는 위에꺼는 여러개의 모델을 구성할때 쓰이며,
# 밑에꺼는 한개의 모델만 구현할때 쓰인다.


# 시퀀셜 함수구성 
# model = Sequential()
# model.add(Dense(3, input_shape=(5,)))
# model.add(Dense(4))
# model.add(Dense(10))
# model.add(Dense(4))
# model.add(Dense(2))

# model.summary()