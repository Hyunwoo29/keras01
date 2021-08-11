from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  #  LogisticRegression 면접질문에서 많이나옴 (분류모델이라고 외우면됨)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split, KFold, cross_val_score
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

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# model = LinearSVC()
# acc:  [0.9125     0.925      0.925      0.97468354 0.92405063] 평균 :  0.9322
# model = SVC()
# acc:  [0.925      0.9        0.9375     0.93670886 0.89873418] 평균 :  0.9196
# model = KNeighborsClassifier()
# acc:  [0.9375     0.925      0.95       0.98734177 0.92405063] 평균 :  0.9448
# model = LogisticRegression()
# acc:  [0.9        0.975      0.95       0.98734177 0.91139241] 평균 :  0.9447
# model = DecisionTreeClassifier()
# acc:  [0.8875     0.925      0.9125     0.93670886 0.87341772] 평균 :  0.907
# model = DecisionTreeRegressor()
# acc:  [0.58736299 0.67545639 0.54415954 0.69379845 0.38105413] 평균 :  0.5764
# model = RandomForestClassifier()
# acc:  [0.95       0.9875     0.9375     0.97468354 0.94936709] 평균 :  0.9598
model = RandomForestRegressor()
# acc:  [0.87131399 0.84182826 0.78950427 0.91254373 0.74756574] 평균 :  0.8326


scores = cross_val_score(model, x_train, y_train, cv=kfold)
print("acc: " ,scores, "평균 : ",round(np.mean(scores), 4))
