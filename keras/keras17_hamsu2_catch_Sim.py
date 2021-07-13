# 06_R2_2를 카피
# 함수형으로 리폼하시오.
# 서머리로 확인 하시오.

from operator import index
from keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from tensorflow.python.keras import activations

x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

optimizers = ['adam']
activations = ['elu', 'selu', 'relu']

r2score = []
opt = []
act = []
y_pred = []

# model = Sequential()
for i in range(len(optimizers)) :
    for j in range(len(activations)) :
        input1 = Input(shape=(1,))
        dense1 = Dense(20)(input1)
        dense2 = Dense(30)(dense1)
        dense3 = Dense(20)(dense2)
        output1 = Dense(1)(dense3)
        model = Model(inputs=input1, outputs=output1)
        # model.add(Dense(100, input_dim = 1))
        # model.add(Dense((1+i)*20))
        # model.add(Dense((1+i)*10,activation=activations[j]))
        # model.add(Dense((1+i)*5,activation=activations[j]))
        # model.add(Dense(1))
        model.compile(loss = 'mse', optimizer=optimizers[i])
        model.fit(x,y,epochs=1000, batch_size=128)

        loss = model.evaluate(x,y)

        y_predict = model.predict(x)
        y_pred.append(y_predict)

        from sklearn.metrics import r2_score
        r2 = r2_score(y,y_predict)
        r2score.append(r2)
        opt.append(optimizers[i])
        act.append(activations[j])

# index = r2score.index(max(r2score))
# print("x의값 : ", y_pred[index])
# print("r2score : ", r2score[index])
model.summary()
'''
x의값 :  [[1.0000002]
 [2.0000007]
 [3.9999957]
 [3.0000012]
 [5.000002 ]]
r2score :  0.9999999999975955
'''


# 과제 2
# R2를 심수안씨를 이겨라!!
# 일 밤 12시까지