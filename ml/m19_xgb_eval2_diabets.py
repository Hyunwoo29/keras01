from os import scandir
from xgboost import XGBRegressor
from sklearn.datasets import load_boston, load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1.데이터
datasets = load_diabetes()
x = datasets["data"]
y = datasets["target"]

# print(x.shape, y.shape)  (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    shuffle=True, random_state=66, train_size=0.8)

scalar = MinMaxScaler()
scalar.fit(x_train)
x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)

model = XGBRegressor(n_estimators=100, learing_rate=0.1, n_jobs=1)

model.fit(x_train,y_train, verbose=1, eval_metric=['rmse', 'mae', 'logloss'],
            eval_set=[(x_train, y_train), (x_test, y_test)]
)

results = model.score(x_test, y_test)
print("results = ", results)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 = ", r2)