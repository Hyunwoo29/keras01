import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, GRU

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]]) # (4, 3)
y = np.array([4,5,6,7]) # (4,)

print(x.shape, y.shape)

x = x.reshape(4, 3, 1) # (batch_size, timesteps, feature)
print(x)
#2. 모델구성
model = Sequential()
model.add(GRU(units=10, activation='relu', input_shape=(3, 1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1))

model.summary()

# Layer (type)                 Output Shape              Param #
# =================================================================
# gru (GRU)                    (None, 10)                390
# _________________________________________________________________
# dense (Dense)                (None, 64)                704
# _________________________________________________________________
# dense_1 (Dense)              (None, 32)                2080
# _________________________________________________________________
# dropout (Dropout)            (None, 32)                0
# _________________________________________________________________
# dense_2 (Dense)              (None, 32)                1056
# _________________________________________________________________
# dense_3 (Dense)              (None, 12)                396
# _________________________________________________________________
# dense_4 (Dense)              (None, 1)                 13
# =================================================================
# Total params: 4,639
# Trainable params: 4,639
# Non-trainable params: 0
# _________________________________________________________________
#GRU Param 490개인 이유를 찾으시오.
#(Input + baias + output + resetgate) * output
# GRU는 데이터가 많을수록 LSTM보다 성능이 떨어진다. 
# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# model.fit(x, y, epochs=100, batch_size=1)

# #4. 평가, 예측
# x_input = np.array([5,6,7]).reshape(1,3,1)
# results = model.predict(np.array(x_input))
# print(results)

