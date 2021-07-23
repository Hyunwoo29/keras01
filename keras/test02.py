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
#1. 데이터
datasets = pd.read_csv('./_data/삼성전자 주가 20210721.csv', sep=',', index_col=0, header=0 , encoding='CP949')
datasets1 = pd.read_csv('./_data/SK주가 20210721.csv', sep=',', index_col=0, header=0, encoding='CP949')
datasets = datasets.sort_values(by='일자')
datasets1 = datasets1.sort_values(by='일자')
datasets.drop(['Unnamed: 15'], axis = 1, inplace = True) # Unnamed: 15 라는 열 제거 
datasets1.drop(['Unnamed: 15'], axis = 1, inplace = True) # Unnamed: 15 라는 열 제거z
datasets= datasets.dropna(axis=0)
datasets1= datasets1.dropna(axis=0)

# print(datasets) # [3601 rows x 0 columns], [3601 rows x 15 columns] ---> 결측치 제거한후 [3600 rows x 14 columns]
# print(datasets1) # [3601 rows x 0 columns] , [3601 rows x 15 columns]    -> 결측치 제거한후 [3600 rows x 14 columns]
# print(datasets.info()) # dtypes: float64(5), object(9)
# print(datasets.isnull().sum()) # Unnamed: 15    3601
# print(datasets1.isnull().sum()) # Unnamed: 15    3601
samsung_df = pd.DataFrame(datasets)
sk_df = pd.DataFrame(datasets1)

samsung = samsung_df[['시가','고가','저가','거래량','종가']]   # 열 추출
sk = sk_df[['시가','고가','저가','거래량','종가']]

samsung = samsung.to_numpy()
sk = sk.to_numpy()

# print(samsung.shape) (3600, 5)

size = 5
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size),:]
        aaa.append(subset)
    return np.array(aaa)

x_samsung = split_x(samsung, size)#(3596, 5, 5)
x_sk = split_x(sk, size)
x_samsung_pred = x_samsung[-1, :] #(5, 5)
x_sk_pred = x_sk[-1, :]
y_samsung = samsung[4:,4] # (3596,)
x_samsung = x_samsung.reshape(3596, 25)
x_sk = x_sk.reshape(3596,25)
y_samsung = y_samsung.reshape(-1, 1)                     
x_samsung_pred = x_samsung_pred.reshape(1,25)
x_sk_pred = x_sk_pred.reshape(1,25)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_samsung, y_samsung, shuffle=False, train_size=0.8)
x1_train, x1_test, y1_train, y1_test = train_test_split(x_sk, y_samsung, shuffle=False, train_size=0.8)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_samsung_pred = scaler.transform(x_samsung_pred)
x_sk_pred = scaler.transform(x_sk_pred)
x1_train = scaler.fit_transform(x1_train)
x1_test = scaler.transform(x1_test)

x_train = x_train.reshape(x_train.shape[0], 5, 5) 
x_test = x_test.reshape(x_test.shape[0], 5, 5)
x_samsung_pred = x_samsung_pred.reshape(x_samsung_pred.shape[0], 5, 5)
x_sk_pred = x_sk_pred.reshape(x_sk_pred.shape[0], 5, 5)
x1_train = x1_train.reshape(x1_train.shape[0], 5, 5)
x1_test = x1_test.reshape(x1_test.shape[0], 5, 5)


input1 = Input(shape=(5, 5))
lstm = LSTM(128, activation='relu', return_sequences=True)(input1)
conv = Conv1D(64, 2, activation='relu')(lstm)
flat = Flatten()(conv)
dense = Dense(48)(flat)
dense = Dense(32)(dense)
output1 = Dense(1)(dense)
# 모델2 - sk
input2 = Input(shape=(5, 5))
lstm = LSTM(128, activation='relu', return_sequences=True)(input2)
conv = Conv1D(64, 2, activation='relu')(lstm)
flat = Flatten()(conv)
dense = Dense(32)(flat)
dense = Dense(16)(dense)
output2 = Dense(1)(dense)
from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1, output2])
merge2 = Dense(10, activation='relu')(merge1)
merge3 = Dense(10, activation='relu')(merge2)
last_output = Dense(1)(merge3)
model = Model(inputs=[input1, input2], outputs=last_output)


model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1, restore_best_weights=True)
model.fit([x_train, x_train], [y_train,y1_train], epochs=1000, batch_size=50, validation_split=0.02, callbacks=[es])
results = model.evaluate([x_test, x1_test], [y_test, y1_test])
print('loss :', results)
y_predict = model.predict([x_samsung_pred, x_sk_pred])
print(y_predict)
