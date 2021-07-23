import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Conv1D, Conv2D, Flatten, MaxPooling1D, MaxPooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D, Dropout, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.backend import concatenate
#1. 데이터
datasets = pd.read_csv('./_data/삼성전자 주가 20210721.csv', sep=',', index_col=0, header=0 , nrows=2601, encoding='CP949')
datasets1 = pd.read_csv('./_data/SK주가 20210721.csv', sep=',', index_col=0, header=0, nrows=2601, encoding='CP949')

datasets = datasets[['시가','고가','저가','거래량','종가']]
datasets1 = datasets1[['시가','고가','저가','거래량','종가']]

datasets = datasets.dropna(axis=0)
datasets1 = datasets.dropna(axis=0)

datasets = datasets.sort_values(by=['일자'], axis=0).values
datasets1 = datasets1.sort_values(by=['일자'], axis=0).values

size = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):  # 10 - 5 + 1 = 6행 // 행의 개수가 정해짐
        subset = dataset[i : (i + size), :]
        aaa.append(subset)
    return np.array(aaa)

samsung = split_x(datasets, size)
sk = split_x(datasets1, size)

x1 = samsung # (3596, 5, 5)
y = samsung[:,size-2,-1:] # (3596, 1)

y = y.reshape(-1) # (3596,)
x2 = sk
x1 = x1.reshape(2597,5 * 5)
x2 = x2.reshape(2597,5 * 5)
x1_pred = samsung[:1,-size:]
x2_pred = sk[:1,-size:]

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.8, random_state=70)

x1_train = x1_train.reshape(x1_train.shape[0], 5, 5)
x1_test = x1_test.reshape(x1_test.shape[0], 5, 5)
x2_train = x2_train.reshape(x2_train.shape[0], 5, 5)
x2_test = x2_test.reshape(x2_test.shape[0], 5, 5)

input1 = Input(shape=(5,5))
x1 = LSTM(128, return_sequences=True, activation='relu')(input1)
x2 = Conv1D(64, 2, activation='relu')(x1)
x3 = Dropout(0.3)(x2)
x4 = Flatten()(x3)
x5 = Dense(128, activation='relu')(x4)
x6 = Dense(64, activation='relu')(x5)
output1 = Dense(32, activation='relu')(x6)



# 모델 2

input2 = Input(shape=(5,5))
x7 = LSTM(128, return_sequences=True, activation='relu')(input2)
x8 = Conv1D(64, 2, activation='relu')(x7)
x9 = Dropout(0.3)(x8)
x10 = Flatten()(x9)
x11 = Dense(128, activation='relu')(x10)
x12 = Dense(64, activation='relu')(x11)
output2 = Dense(32, activation='relu')(x12)

merge = concatenate([output1, output2])
merge1 = Dense(64)(merge)
output = Dense(1)(merge1)

model = Model(inputs=[input1, input2], outputs=output)

model.summary()
import datetime
# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
date = datetime.datetime.now() 
date_time = date.strftime("%m%d_%H%M") 

filepath = './_save/' 
filename = '.{epoch:04d}-{val_loss:4f}.hdf5' 
modelpath = "".join([filepath, "samsung", date_time, "_", filename])
cp = ModelCheckpoint(monitor='val_loss', patience=20, verbose=1, mode='auto', save_best_only=True,
                    filepath= modelpath)
es = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto', restore_best_weights=True)

model.fit([x1_train, x2_train], y_train, epochs=50, batch_size=16, verbose=1, validation_split=0.2, callbacks=[es,cp])


loss = model.evaluate([x1_test, x2_test], y_test)
y_predict = model.predict([x1_pred, x2_pred])



print(loss)
print(y_predict)