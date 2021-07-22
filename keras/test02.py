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
datasets1.drop(['Unnamed: 15'], axis = 1, inplace = True) # Unnamed: 15 라는 열 제거
datasets= datasets.dropna(axis=0)
datasets1= datasets1.dropna(axis=0)

# print(datasets) # [3601 rows x 0 columns], [3601 rows x 15 columns] ---> 결측치 제거한후 [3600 rows x 14 columns]
# print(datasets1) # [3601 rows x 0 columns] , [3601 rows x 15 columns]    -> 결측치 제거한후 [3600 rows x 14 columns]
# print(datasets.info()) # dtypes: float64(5), object(9)
# print(datasets.isnull().sum()) # Unnamed: 15    3601
# print(datasets1.isnull().sum()) # Unnamed: 15    3601

datasets.drop(['종가 단순 5', '10', '20', '60', '120', '단순 5', '20.1', '60.1', '120.1'], axis='columns', inplace=True)
datasets1.drop(['종가 단순 5', '10', '20', '60', '120', '단순 5', '20.1', '60.1', '120.1'], axis='columns', inplace=True)

datasets = datasets.to_numpy()
datasets1 = datasets1.to_numpy()


size = 5
def split_x(datasets, size):
    aaa = []
    for i in range(len(datasets) - size + 1):
        subset = datasets[i : (i + size),:]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(datasets, size)
predset = split_x(datasets1,size)
print(dataset.shape, predset.shape)

# scaler = MinMaxScaler()
# data1 = datasets.iloc[:, :-1]
# data2 = datasets.iloc[:, -1:]
# data1 = scaler.fit_transform(data1)
# data2 = scaler.fit_transform(data2)
# dataset = np.concatenate((data1, data2), axis=1)

# scaler = MinMaxScaler()
# data_1 = datasets1.iloc[:, :-1]
# data_2 = datasets1.iloc[:,-1:]
# data_1 = scaler.fit_transform(data_1)
# data_2 = scaler.fit_transform(data_2)
# dataset1 = np.concatenate((data_1, data_2), axis=1)
# print(dataset.shape, dataset1.shape) # (3600, 5) (3600, 5)



