from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  #  LogisticRegression 면접질문에서 많이나옴 (분류모델이라고 외우면됨)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


import numpy as np
from sklearn import datasets
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
datasets = load_wine()
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape, y.shape)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 

# model = LinearSVC()
# accuracy_score :  0.9814814814814815
# model = SVC()
# accuracy_score :  0.9814814814814815
# model = KNeighborsClassifier()
# accuracy_score :  0.9444444444444444
# model = LogisticRegression()
# accuracy_score :  0.9814814814814815
# model = DecisionTreeClassifier()
# accuracy_score :  0.9629629629629629
# model = RandomForestClassifier()
# accuracy_score :  1.0

model.fit(x_train, y_train)

results = model.score(x_test, y_test) # 머신러닝에서는 evaluate 개념이 score이다.
print("modle.score : ", results)

from sklearn.metrics import r2_score, accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)