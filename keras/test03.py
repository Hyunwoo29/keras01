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
datasets = pd.read_csv('./_data/삼성전자 주가 20210721.csv', sep=',', index_col=0, header=0 ,nrows=2602, encoding='CP949')
datasets1 = pd.read_csv('./_data/SK주가 20210721.csv', sep=',', index_col=0, header=0,nrows=2602, encoding='CP949')
datasets = datasets[['시가','고가','저가','거래량','종가']]
datasets1 = datasets1[['시가','고가','저가','거래량','종가']]

datasets = datasets.dropna(axis=0)
datasets1 = datasets.dropna(axis=0)

datasets = datasets.sort_values(by=['일자'], axis=0).values
datasets1 = datasets1.sort_values(by=['일자'], axis=0).values

# print(datasets) # [3601 rows x 0 columns], [3601 rows x 15 columns] ---> 결측치 제거한후 [3600 rows x 14 columns]
# print(datasets1) # [3601 rows x 0 columns] , [3601 rows x 15 columns]    -> 결측치 제거한후 [3600 rows x 14 columns]
# print(datasets.info()) # dtypes: float64(5), object(9)
# print(datasets.isnull().sum()) # Unnamed: 15    3601
# print(datasets1.isnull().sum()) # Unnamed: 15    3601
datasets = pd.DataFrame(datasets)
datasets1 = pd.DataFrame(datasets1)
data1 = datasets.iloc[:, :-1]
data2 = datasets.iloc[:, -1:]
scaler = MinMaxScaler()
data1 = scaler.fit_transform(data1)
data2 = scaler.fit_transform(data2)
dataset = np.concatenate((data1, data2), axis=1)

data_1 = datasets1.iloc[:, :-1]
data_2 = datasets1.iloc[:,-1:]
scaler = MinMaxScaler()
data_1 = scaler.fit_transform(data_1)
data_2 = scaler.fit_transform(data_2)
dataset1 = np.concatenate((data_1, data_2), axis=1)
print(dataset.shape, dataset1.shape) # (3600, 5) (3600, 5)

ss = dataset[:, [-2]]

x1 = []
x2 = []
y = []
size = 50
for i in range(len(ss) - size + 1):
    x1.append(dataset[i: (i + size) ])
    x2.append(dataset1[i: (i + size) ])
    y.append(ss[i + (size - 1)]) 
# 설정한 단위일수 만큼 최근 데이터 slice -> y_predict 를 위한 x1_pred, x2_pred 생성
x1_pred = [dataset[len(dataset) - size : ]]
x2_pred = [dataset1[len(dataset1) - size : ]]

x1 = np.array(x1)
x2 = np.array(x2)
y = np.array(y)
x1_pred = np.array(x1_pred)
x2_pred = np.array(x2_pred)
print(x1.shape, x2.shape, y.shape, x1_pred.shape, x2_pred.shape) # (2553, 50, 5) (2553, 50, 5) (2553, 1) (1, 50, 5) (1, 50, 5)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, test_size=0.2, random_state=66)

#2. Modeling
input1 = Input(shape=(50, 5))
d1 = LSTM(32, activation='relu')(input1)
d2 = Dense(16, activation='relu')(d1)

input2 = Input(shape=(50, 5))
d3 = LSTM(32, activation='relu')(input2)
d4 = Dense(16, activation='relu')(d3)
from tensorflow.keras.layers import concatenate
m = concatenate([d2, d4])
output = Dense(1, activation='relu')(m)

model = Model(inputs=[input1, input2], outputs=output)

#3. Compiling, Training
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', restore_best_weights=False, mode='auto', verbose=1, patience=4)

model.fit([x1_train, x2_train], y_train, epochs=4, batch_size=16, verbose=1, validation_split=0.001, callbacks=[es])


#4. Evaluating, Prediction
loss = model.evaluate([x1_test, x2_test], y_test)
y_pred = model.predict([x1_pred, x2_pred])
y_pred = scaler.inverse_transform(y_pred) # 가격(원) 확인을 위한 inverse scaling

print('loss = ', loss)
print("Tomorrow's closing price = ", y_pred)


