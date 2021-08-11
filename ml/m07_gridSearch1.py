import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression #  LogisticRegression 면접질문에서 많이나옴 (분류모델이라고 외우면됨)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score

datasets = load_iris()

x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)


parameters = [
    {"C" : [1,10,100,1000], "kernel":["linear"]},   # kernel => activation function
    {"C" : [1,10,100], "kernel":["rbf"] , "gamma": [0.001,0.0001] }, # gamma => lr function
    {"C" : [1,10,100,1000],"kernel":["sigmoid"], "gamma":[0.001,0.0001] }
]

# model = SVC()

model = GridSearchCV(SVC(), parameters, cv=kfold)

model.fit(x_train, y_train)

print("최적의 매개변수 : ", model.best_estimator_)
print("best_score : ", model.best_score_)
print("model.score : ", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("정답률 : ", accuracy_score(y_test, y_predict))

# 최적의 매개변수 :  SVC(C=1, kernel='linear')
# best_score :  0.9916666666666668
# model.score :  0.9666666666666667
# 정답률 :  0.9666666666666667