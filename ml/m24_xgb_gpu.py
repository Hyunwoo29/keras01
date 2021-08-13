from os import scandir
from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1.데이터
datasets = load_boston()
x = datasets["data"]
y = datasets["target"]

# print(x.shape, y.shape)  (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    shuffle=True, random_state=66, train_size=0.8)

scalar = MinMaxScaler()
scalar.fit(x_train)
x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)

model = XGBRegressor(n_estimators=10000, learing_rate=0.01, # n_jobs= -1
                    tree_method='gpu_hist',
                    predictor='gpu_predictor', # cpu_predictor    cpu로 돌릴지 gpu로 돌릴지 결정
                    gpu_id=0
)
import time
start_time = time.time()
model.fit(x_train,y_train, verbose=1, eval_metric='rmse', # 'mae', 'logloss'],
            eval_set=[(x_train, y_train), (x_test, y_test)]
)

print("걸린시간 : ", time.time()- start_time)


# njobs = 1 걸린시간 : 3.717205047607422
# njobs=-1 걸린시간 :  4.750133037567139

# njobs=2 걸린시간 : 3.2404072284698486
# njobs=4 걸린시간 :  3.1331167221069336

# njobs=8 걸린시간 :  3.3341634273529053

# tree_method='gpu_hist'   걸린시간 :  18.531525373458862
# gpu_id=0 걸린시간 :  19.620315551757812