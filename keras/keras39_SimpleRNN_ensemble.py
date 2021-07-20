# 앙상블 모델을 만드시오.
# 결과치 신경쓰지 말고 모델만 완성할것!
# 실습
# keras37을 함수형으로 리폼하시오.
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU, Input
from tensorflow.python import keras

#1. 데이터
x1 = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], 
            [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])
x2 = np.array([[10,20,30], [20,30,40], [30,40,50], [40,50,60], [50,60,70],
            [60,70,80], [70,80,90], [80,90,100], 
            [90,100,110], [100,110,120],
            [2,3,4], [3,4,5], [4,5,6]])  
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x1_pred = np.array([55,65,75])   # (3,)
x2_pred = np.array([65,75,85])   # (3,)

print(x1.shape)      # (13,3)
print(y.shape)      # (13,)

x1 = x1.reshape(x1.shape[0],x1.shape[1],1)
x2 = x2.reshape(x2.shape[0],x2.shape[1],1)

#2. 모델구성
input1 = Input(shape=(3,1))
simplernn1 = SimpleRNN(5, activation='relu')(input1)
dense1 = Dense(18, activation='relu')(simplernn1)
dense1 = Dense(10, activation='relu')(dense1)
dense1 = Dense(12, activation='relu')(dense1)

input2 = Input(shape=(3,1))
simplernn2 = SimpleRNN(5, activation='relu')(input2)
dense2 = Dense(10, activation='relu')(simplernn2)
dense2 = Dense(8, activation='relu')(dense2)
dense2 = Dense(12, activation='relu')(dense2)

from tensorflow.keras.layers import concatenate, Concatenate
merge1 = concatenate([dense1, dense2])
middle = Dense(10, activation= 'relu') (merge1)
middle = Dense(30, activation= 'relu') (middle)

output1 = Dense(10, activation= 'relu') (middle)
output1 = Dense(40, activation= 'relu') (output1)
output1 = Dense(20, activation= 'relu') (output1)
output1 = Dense(1, activation= 'relu') (output1)

model = Model(inputs = [input1,input2], outputs = [output1])
# model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit([x1,x2], y, epochs=100, batch_size=8)

#4. 평가, 예측
x1_pred = x1_pred.reshape(1,3,1)
x2_pred = x2_pred.reshape(1,3,1)
results = model.predict([x1_pred,x2_pred])
print(results)

# 80에 가까이 예측
# [[76.34838]]
