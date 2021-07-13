from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from tensorflow.python.keras.layers.core import Dropout

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (442, 10) (442,)

# print(datasets.feature_names) # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# print(datasets.DESCR)

# print(y[:30])
# print(np.min(y), np.max(y))
#2. 모델구성
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=110)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                    train_size=0.8,
                                                    shuffle=True)
optimizers = ['adam']
activations = ['elu', 'relu', 'linear']

r2score = []
opt = []
act = []
y_pred = []
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=60, mode='min')


# model = Sequential()
for i in range(len(optimizers)) :
    for j in range(len(activations)) :
        input1 = Input(shape=(10,))
        dense1 = Dense(20)(input1)
        dense2 = Dense(30)(dense1)
        dense3 = Dense(20)(dense2)
        output1 = Dense(1)(dense3)
        model = Model(inputs=input1, outputs=output1)
        model.compile(loss = 'mae', optimizer=optimizers[i])
        model.fit(x_train, y_train, epochs = 500, batch_size= 32, validation_data=(x_val, y_val),callbacks=[early_stopping])

        loss = model.evaluate(x_test, y_test)

        y_predict = model.predict(x_test)
        y_pred.append(y_predict)

        from sklearn.metrics import r2_score
        r2 = r2_score(y_test, y_predict)
        r2score.append(r2)
        opt.append(optimizers[i])
        act.append(activations[j])

model.summary()
# print("loss : ", loss)
# y_predict = model.predict(x_test)
# index = r2score.index(max(r2score))
# print("x의값 : ", y_pred[index])
# print("best optimizer : ",opt[index],", best activation : ",act[index],", r2score : ", r2score[index])

#3. 컴파일, 훈련

#4. 평가, 예측
# r2score :  0.6010348369308107
# 0.6156926546917119
# r2score :  0.6204118565460017
#  best activation :  elu , r2score :  0.6267166546698073