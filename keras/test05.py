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

samsung = samsung_df[['시가','고가','저가','종가','거래량']]   # 열 추출
sk = sk_df[['시가','고가','저가','종가','거래량']]


data1 = samsung.iloc[:, 0:4]
data2 = samsung.iloc[:, -1:]
scaler = MinMaxScaler()
data1 = scaler.fit_transform(data1)
data2 = scaler.fit_transform(data2)
dataset = np.concatenate((data1, data2), axis=1)

data_1 = sk.iloc[:, 0:4]
data_2 = sk.iloc[:,-1:]
scaler = MinMaxScaler()
data_1 = scaler.fit_transform(data_1)
data_2 = scaler.fit_transform(data_2)
dataset1 = np.concatenate((data_1, data_2), axis=1)
# print(dataset.shape, dataset1.shape) # (3600, 5) (3600, 5)
nam = ['시가','고가','저가','종가','거래량']
samsung = pd.DataFrame(dataset,   columns=nam)
sk = pd.DataFrame(dataset1, columns=nam)
samsung_1 = samsung[['고가','저가','종가','거래량','시가']] 
sk_1 = sk[['고가','저가','종가','거래량','시가']]

##데이터셋 생성
y = samsung_1.iloc[6:,-1] # 타겟은 주식 시가

samsung_1 =samsung_1.to_numpy()
sk_1 = sk_1.to_numpy()
y = y.to_numpy()
print(samsung_1.shape, y.shape)


size = 5

def split_x(samsung_1, size):
    aaa = []
    for i in range(len(samsung_1) - size + 1):  # 10 - 5 + 1 = 6행 // 행의 개수가 정해짐
        subset = samsung_1[i : (i + size), :]
        aaa.append(subset)
    return np.array(aaa)

size1 = 1
def split_y(y, size1):
    aa = []
    for i in range(len(y) - size1 + 1):  # 10 - 5 + 1 = 6행 // 행의 개수가 정해짐
        subset = y[i : (i + size1)]
        aa.append(subset)
    return np.array(aa)

def split_x1(sk_1, size):
    a = []
    for i in range(len(sk_1) - size + 1):  # 10 - 5 + 1 = 6행 // 행의 개수가 정해짐
        subset = sk_1[i : (i + size), :]
        a.append(subset)
    return np.array(a)

samsung = split_x(samsung_1, size)
sk = split_x1(sk_1, size)
y = split_y(y, size1)

# print(samsung.shape, sk.shape, y.shape)  # (3596, 5, 5) (3596, 5, 5) (3594, 1)
y = np.array(y)
print(y.shape)

x_train, x_test, x1_train, x1_test, y_train, y_test = train_test_split(samsung, sk, y, test_size=0.2, random_state=60)
# # x_train = x_train.reshape(x_train.shape[0], 5, 5)
# # x_test = x_test.reshape(x_test.shape[0], 5, 5)
# # x1_train = x1_train.reshape(x1_train.shape[0], 5, 5)
# # x1_test = x1_test.reshape(x1_test.shape[0], 5, 5)
# # y_test = y_test.reshape(y_test.shape[0], 1, 1)

# print(y_train.shape)