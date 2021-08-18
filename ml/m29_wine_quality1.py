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

datasets = datasets.values
# ic(datasets)
# ic(type(datasets)) <class 'numpy.ndarray'>
# ic(datasets.shape) (4898, 12)

x = datasets[:, :11]
y = datasets[:, 11]

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, shuffle=True, train_size=0.8
)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2.모델
model = XGBClassifier(n_jobs=-1)

#3.훈련
model.fit(x_train, y_train)

#4. 평가, 예측
score = model.score(x_test, y_test)

ic("accuracy : ", score)
# ic| 'accuracy : ', score: 0.6816326530612244