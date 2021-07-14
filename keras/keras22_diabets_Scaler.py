#실습 diabets
# 1. loss와 r2로 평가를 함
# 2. MinMax와 Standard 결과를 명시

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)
# 데이터 전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train) # train은 훈련을 시키고, test는 훈련에 포함되면 안된다.
x_test = scaler.transform(x_test) 

# print(x.shape, y.shape) # (442, 10) (442,)

# print(datasets.feature_names) # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# print(datasets.DESCR)

# print(y[:30])
# print(np.min(y), np.max(y))
#2. 모델구성


#모델구성
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(10,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=8)

loss = model.evaluate(x_test, y_test)
print('lose : ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2socre: ', r2)

# MaxAbsScaler
# lose :  50.73676681518555
# r2socre:  0.25601042189826395

# lose :  52.6964111328125
# r2socre:  0.27823540314170714

# lose :  48.845272064208984
# r2socre:  0.31781112780014564

# RobustScaler
# lose :  52.739013671875
# r2socre:  0.2433266475874858

# lose :  51.82915115356445
# r2socre:  0.24872060302911003

# QuantileTransformer
# lose :  46.422889709472656
# r2socre:  0.4306010606475259

# lose :  45.10093307495117
# r2socre:  0.4554468928476876

# PowerTransformer
# lose :  55.895118713378906
# r2socre:  0.19138549114993886

# lose :  54.16565704345703
# r2socre:  0.2257124494076037