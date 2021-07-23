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

samsung = samsung_df[['시가','고가','저가','거래량','종가']]   # 열 추출
sk = sk_df[['시가','고가','저가','거래량','종가']]

high_prices = max(samsung['종가'].values)
low_prices = min(samsung['종가'].values)
mid_prices = (high_prices + low_prices)/2

samsung = samsung.to_numpy()
sk = sk.to_numpy()

seq_len = 50
seqeunce_length = seq_len + 1

result = []
for index in range(len(mid_prices) - seqeunce_length):
    result.append(mid_prices[index: index + seqeunce_length])

normalize_data = []
for window in result:
    normalized_window = [((float(p)/ float(window[0])) -1)for p in window]
    normalize_data.append(normalized_window)

result = np.array(normalize_data)


