from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  #  LogisticRegression 면접질문에서 많이나옴 (분류모델이라고 외우면됨)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier


import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

datasets = load_breast_cancer()

# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (569, 30) (569,)

# print(y[:20]) # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
# print(np.unique(y)) # [0 1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 

# model = LinearSVC()
# accuracy_score :  0.9766081871345029
# model = SVC()
# accuracy_score :  0.9766081871345029
# model = KNeighborsClassifier()
# accuracy_score :  0.9590643274853801
# model = LogisticRegression()
# accuracy_score :  0.9824561403508771
# model = DecisionTreeRegressor()
# accuracy_score :  0.9415204678362573
# model = RandomForestClassifier()
# accuracy_score :  0.9707602339181286
model.fit(x_train,y_train)

results = model.score(x_test, y_test) # 머신러닝에서는 evaluate 개념이 score이다.
print("modle.score : ", results)

from sklearn.metrics import r2_score, accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)


