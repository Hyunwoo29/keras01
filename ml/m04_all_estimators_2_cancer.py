from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  #  LogisticRegression 면접질문에서 많이나옴 (분류모델이라고 외우면됨)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import all_estimators

import warnings
warnings.filterwarnings('ignore')

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
AdaBoostClassifier 의 정답률 :  0.7961251862891208
BaggingClassifier 의 정답률 :  0.7706408345752608
BernoulliNB 의 정답률 :  0.7451564828614009
CalibratedClassifierCV 의 정답률 :  0.8470938897168405
CategoricalNB 은 null
ClassifierChain 은 null
ComplementNB 은 null
DecisionTreeClassifier 의 정답률 :  0.7706408345752608
DummyClassifier 의 정답률 :  -0.5545454545454545
ExtraTreeClassifier 의 정답률 :  0.7706408345752608
ExtraTreesClassifier 의 정답률 :  0.8725782414307004
GaussianNB 의 정답률 :  0.7706408345752608
GaussianProcessClassifier 의 정답률 :  0.8470938897168405
GradientBoostingClassifier 의 정답률 :  0.8470938897168405
HistGradientBoostingClassifier 의 정답률 :  0.8725782414307004
KNeighborsClassifier 의 정답률 :  0.8216095380029806
LabelPropagation 의 정답률 :  0.7706408345752608
LabelSpreading 의 정답률 :  0.7706408345752608
LinearDiscriminantAnalysis 의 정답률 :  0.8470938897168405
LinearSVC 의 정답률 :  0.8980625931445604
LogisticRegression 의 정답률 :  0.9235469448584203
LogisticRegressionCV 의 정답률 :  0.9490312965722801
MLPClassifier 의 정답률 :  0.8725782414307004
MultiOutputClassifier 은 null
MultinomialNB 은 null
NearestCentroid 의 정답률 :  0.7451564828614009
NuSVC 의 정답률 :  0.7706408345752608
OneVsOneClassifier 은 null
OneVsRestClassifier 은 null
OutputCodeClassifier 은 null
PassiveAggressiveClassifier 의 정답률 :  0.7961251862891208
Perceptron 의 정답률 :  0.9235469448584203
QuadraticDiscriminantAnalysis 의 정답률 :  0.7706408345752608
RadiusNeighborsClassifier 은 null
RandomForestClassifier 의 정답률 :  0.8216095380029806
RidgeClassifier 의 정답률 :  0.8216095380029806
RidgeClassifierCV 의 정답률 :  0.8216095380029806
SGDClassifier 의 정답률 :  0.8470938897168405
SVC 의 정답률 :  0.8980625931445604
StackingClassifier 은 null
VotingClassifier 은 null
'''





