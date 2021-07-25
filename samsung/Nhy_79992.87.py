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
from tensorflow.python.keras.utils.np_utils import normalize
#1. 데이터
datasets = pd.read_csv('./_data/삼성전자 주가 20210721.csv', sep=',', header=0 ,usecols=[1,2,3,4,10], encoding='CP949')
datasets1 = pd.read_csv('./_data/SK주가 20210721.csv', sep=',', index_col=0,usecols=[1,2,3,4,10], header=0, encoding='CP949')
# datasets = datasets.sort_values(by='일자')
# datasets1 = datasets1.sort_values(by='일자')
# datasets.drop(['Unnamed: 15'], axis = 1, inplace = True) # Unnamed: 15 라는 열 제거 
# datasets1.drop(['Unnamed: 15'], axis = 1, inplace = True) # Unnamed: 15 라는 열 제거z
# datasets= datasets.dropna(axis=0)
# datasets1= datasets1.dropna(axis=0)
datasets1 = datasets1[::-1]
datasets = datasets[::-1]
datasets1 = datasets1.dropna(axis=0)
datasets = datasets.dropna(axis=0)

# print(datasets) # [3601 rows x 0 columns], [3601 rows x 15 columns] ---> 결측치 제거한후 [3600 rows x 14 columns]
# print(datasets1) # [3601 rows x 0 columns] , [3601 rows x 15 columns]    -> 결측치 제거한후 [3600 rows x 14 columns]
# print(datasets.info()) # dtypes: float64(5), object(9)
# print(datasets.isnull().sum()) # Unnamed: 15    3601
# print(datasets1.isnull().sum()) # Unnamed: 15    3601
# samsung_df = pd.DataFrame(datasets)
# sk_df = pd.DataFrame(datasets1)

# samsung = samsung_df[['시가','고가','저가','종가','거래량']]   # 열 추출
# sk = sk_df[['시가','고가','저가','종가','거래량']]


data1 = datasets.iloc[:, 0:4]
data2 = datasets.iloc[:, -1:]
scaler = MinMaxScaler()
scaler.fit(data1)
data11 = scaler.transform(data1)
scaler.fit(data2)
data12 = scaler.transform(data2)
dataset = np.concatenate((data11, data12), axis=1)
scaled = np.max(data1) - np.min(data1) 
# print(scaled, min(data1[0]) )
data_1 = datasets1.iloc[:, 0:4]
data_2 = datasets1.iloc[:,-1:]
scaler = MinMaxScaler()
data_1 = scaler.fit_transform(data_1)
data_2 = scaler.fit_transform(data_2)
dataset1 = np.concatenate((data_1, data_2), axis=1)
# print(dataset.shape, dataset1.shape) # (3600, 5) (3600, 5)
# nam = ['시가','고가','저가','종가','거래량']
# samsung = pd.DataFrame(dataset,   columns=nam)
# sk = pd.DataFrame(dataset1, columns=nam)
# samsung_1 = samsung[['고가','저가','종가','거래량','시가']] 
# sk_1 = sk[['고가','저가','종가','거래량','시가']]

##데이터셋 생성
y = dataset[0:,[-5]] # 타겟은 주식 시가

x1 = [] # 삼성
x2 = [] # SK
y1 = [] # target
size = 50 # 데이터 slice 단위일수 설정
for i in range(len(y) - size + 1):
    x1.append(dataset[i: (i + size) ]) # 단위일수만큼 끊어서 저장
    x2.append(dataset1[i: (i + size) ]) # 단위일수만큼 끊어서 저장
    y1.append(y[i + (size - 3)]) # 3일 후의 주가 훈련 및 예측
# y_predict 를 위한 x1_pred, x2_pred 생성
x1_pred = [dataset[len(dataset) - size : ]]
x2_pred = [dataset1[len(dataset1) - size : ]]


# numpy 배열화
x1 = np.array(x1)
x2 = np.array(x2)
y1 = np.array(y1)
x1_pred = np.array(x1_pred)
x2_pred = np.array(x2_pred)
print(x1.shape, x2.shape, y1.shape, x1_pred.shape, x2_pred.shape) # (3551, 50, 5) (3551, 50, 5) (3551,) (1, 50, 5) (1, 50, 5)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y1, test_size=0.2, random_state=66)

#2. Modeling
input1 = Input(shape=(50, 5))
xx = LSTM(16, return_sequences=True,activation='relu')(input1)
xx1 = Conv1D(64, 2, activation='relu')(xx)
xx2 = Dropout(0.2)(xx1)
xx3 = Flatten()(xx2)
xx4 = Dense(64, activation='relu')(xx3)
output1 = Dense(32, activation='relu')(xx4)

input2 = Input(shape=(50, 5))
xy = LSTM(16,return_sequences=True, activation='relu')(input2)
xy1 = Conv1D(64, 2, activation='relu')(xy)
xy2 = Dropout(0.2)(xy1)
xy3 = Flatten()(xy2)
xy4 = Dense(64, activation='relu')(xy3)
output2 = Dense(32, activation='relu')(xy4)

from tensorflow.keras.layers import concatenate
merge = concatenate([output1, output2])
merge1 = Dense(16)(merge)
output = Dense(1)(merge1)
model = Model(inputs=[input1, input2], outputs=output)

#3. Compiling, Training
import datetime
model.compile(loss='mse', optimizer='adam')
date = datetime.datetime.now()
date_time = date.strftime('%m%d_%H%M')
path = './_save/mcp/'
info = '{epoch:02d}_{val_loss:.4f}'
filepath = ''.join([path, 'Nhwoo', '_', date_time, '_', info, '.hdf5'])
mcp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', verbose=1, filepath=filepath)
es = EarlyStopping(monitor='val_loss', restore_best_weights=False, mode='auto', verbose=1, patience=4)
import time
start_time = time.time()
model.fit([x1_train, x2_train], y_train, epochs=5, batch_size=6, verbose=1, validation_split=0.2, callbacks=[es,mcp])
end_time = time.time() - start_time

#4. Evaluating, Prediction
loss = model.evaluate([x1_test, x2_test], y_test)
y_pred = model.predict([x1_pred, x2_pred])
y_pred = y_pred * scaled[0] + np.min(data1)[0]
# y_pred = scaler.inverse_transform(y_pred)

print('loss = ', loss)
print("월요일 삼성 시가 = ", y_pred)
print('걸린시간 : ', end_time)

# loss =  0.00025144265964627266
# 월요일 삼성 시가 =  [[82165.055]]
# 걸린시간 :  116.45184588432312

# loss =  0.0004519543144851923
# 월요일 삼성 시가 =  [[77812.98]]
# 걸린시간 :  295.428094625473

# 2021-07-26 03:15:49.589066: I tensorflow/core/platform/windows/subprocess.cc:308] SubProcess ended with return code: 4294967295
# loss =  0.00016916989989113063
# 월요일 삼성 시가 =  [[78652.42]]
# 걸린시간 :  127.15888333320618

# 2021-07-26 03:24:17.935105: I tensorflow/core/platform/windows/subprocess.cc:308] SubProcess ended with return code: 4294967295
# loss =  0.00014769850531592965
# 월요일 삼성 시가 =  [[79992.87]]
# 걸린시간 :  308.83906865119934

