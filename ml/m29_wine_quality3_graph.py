import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from icecream import ic
from xgboost import XGBClassifier

datasets = pd.read_csv('./_data/winequality-white.csv',
                        index_col=None, header=0, sep=';')

# print(datasets.head())
# print(datasets.shape) (4898, 12)
# ic(datasets.describe())

# datasets = datasets.values
# # ic(datasets)
# # ic(type(datasets)) <class 'numpy.ndarray'>
# # ic(datasets.shape) (4898, 12)

# x = datasets[:, :11]
# y = datasets[:, 11]

# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, random_state=66, shuffle=True, train_size=0.8
# )

# y데이터의 라벨당 갯수를 바 그래프로 그리시오!
count_data = datasets.groupby('quality')['quality'].count()
import matplotlib.pyplot as plt
# count_data.plot()
plt.bar(count_data.index, count_data)
plt.show()


# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# def outliers(data_out):
#     allout = []
#     for i in range(data_out.shape[1]):
#         quartile_1, q2, quartile_3 = np.percentile(data_out[:,i], [25, 50, 75])
#         print('1사분위(25%지점): ',  quartile_1)
#         print('q2(50%지점): ',  q2)
#         print('3사분위(75%지점): ',  quartile_3)
#         iqr = quartile_3 - quartile_1   # IQR(InterQuartile Range, 사분범위)
#         print('iqr: ', iqr)
#         lower_bound = quartile_1 - (iqr * 1.5)  # 하계
#         upper_bound = quartile_3 + (iqr * 1.5)  # 상계
#         print('lower_bound: ', lower_bound)
#         print('upper_bound: ', upper_bound)

#         a = np.where((data_out[:,i]>upper_bound) | (data_out[:,i]<lower_bound)) 
#         allout.append(a)

#     return np.array(allout)


# outlier_loc = outliers(x_train)

# print('이상치의 위치: ', outlier_loc)
# # 아웃라이어의 갯수를 count 하는 기능 추가할것!

# #2.모델
# model = XGBClassifier(n_jobs=-1)

# #3.훈련
# model.fit(x_train, y_train)

# #4. 평가, 예측
# score = model.score(x_test, y_test)

# ic("accuracy : ", score)
# # ic| 'accuracy : ', score: 0.6816326530612244