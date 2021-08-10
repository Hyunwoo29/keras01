from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  #  LogisticRegression 면접질문에서 많이나옴 (분류모델이라고 외우면됨)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.utils import all_estimators

import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score

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
allAlgorithms = all_estimators(type_filter='classifier')

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 



for(name, algorithm) in allAlgorithms :
    try:    
        model = algorithm()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 : ', r2_score(y_test, y_pred))
    except :
        print(name, '은 null')

'''
AdaBoostClassifier 의 정답률 :  0.14772727272727282
BaggingClassifier 의 정답률 :  0.9659090909090909
BernoulliNB 의 정답률 :  0.8977272727272727
CalibratedClassifierCV 의 정답률 :  0.9659090909090909
CategoricalNB 은 null
ClassifierChain 은 null
ComplementNB 은 null
DecisionTreeClassifier 의 정답률 :  0.7954545454545454
DummyClassifier 의 정답률 :  -0.09090909090909083
ExtraTreeClassifier 의 정답률 :  0.625
ExtraTreesClassifier 의 정답률 :  0.9318181818181819
GaussianNB 의 정답률 :  0.9659090909090909
GaussianProcessClassifier 의 정답률 :  0.9318181818181819
GradientBoostingClassifier 의 정답률 :  0.9318181818181819
HistGradientBoostingClassifier 의 정답률 :  1.0
KNeighborsClassifier 의 정답률 :  0.8977272727272727
LabelPropagation 의 정답률 :  0.9318181818181819
LabelSpreading 의 정답률 :  0.9318181818181819
LinearDiscriminantAnalysis 의 정답률 :  0.9659090909090909
LinearSVC 의 정답률 :  0.9659090909090909
LogisticRegression 의 정답률 :  0.9659090909090909
LogisticRegressionCV 의 정답률 :  0.9659090909090909
MLPClassifier 의 정답률 :  1.0
MultiOutputClassifier 은 null
MultinomialNB 은 null
NearestCentroid 의 정답률 :  0.9318181818181819
NuSVC 의 정답률 :  0.9659090909090909
OneVsOneClassifier 은 null
OneVsRestClassifier 은 null
OutputCodeClassifier 은 null
PassiveAggressiveClassifier 의 정답률 :  0.9659090909090909
Perceptron 의 정답률 :  0.8977272727272727
QuadraticDiscriminantAnalysis 의 정답률 :  0.9318181818181819
RadiusNeighborsClassifier 은 null
RandomForestClassifier 의 정답률 :  1.0
RidgeClassifier 의 정답률 :  0.9318181818181819
RidgeClassifierCV 의 정답률 :  0.9659090909090909
SGDClassifier 의 정답률 :  1.0
SVC 의 정답률 :  0.9659090909090909
StackingClassifier 은 null
VotingClassifier 은 null
'''


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

# model.fit(x_train, y_train)

# results = model.score(x_test, y_test) # 머신러닝에서는 evaluate 개념이 score이다.
# print("modle.score : ", results)

# from sklearn.metrics import r2_score, accuracy_score
# y_predict = model.predict(x_test)
# acc = accuracy_score(y_test, y_predict)
# print("accuracy_score : ", acc)