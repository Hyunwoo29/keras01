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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
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

# MinMaxScaler
# lose :  46.34684753417969
# r2socre:  0.4850533704918506

# lose :  46.05620193481445
# r2socre:  0.4882984921833149

# lose :  45.13606262207031
# r2socre:  0.5010672874418272

# lose :  45.265628814697266
# r2socre:  0.5016187165553718

# StandardScaler
# lose :  52.15022277832031
# r2socre:  0.25362154989597885

# lose :  53.877830505371094
# r2socre:  0.2661553292981409

# lose :  51.24164581298828
# r2socre:  0.2994498846113357
