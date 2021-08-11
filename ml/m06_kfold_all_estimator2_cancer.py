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
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score

datasets = load_breast_cancer()


x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)
allAlgorithms = all_estimators(type_filter='classifier')
kfold = KFold(n_splits=5, shuffle=True, random_state=66)

for(name, algorithm) in allAlgorithms :
    try:    
        model = algorithm()
        scores = cross_val_score(model, x, y, cv=kfold)
        print(name, scores, round(np.mean(scores), 4))
    except :
        print(name, '은 null')

'''
AdaBoostClassifier [0.94736842 0.99122807 0.94736842 0.96491228 0.97345133] 0.9649
BaggingClassifier [0.93859649 0.92982456 0.96491228 0.92982456 0.95575221] 0.9438
BernoulliNB [0.64035088 0.65789474 0.62280702 0.5877193  0.62831858] 0.6274    
CalibratedClassifierCV [0.89473684 0.93859649 0.89473684 0.92982456 0.97345133] 0.9263
CategoricalNB [nan nan nan nan nan] nan
ClassifierChain 은 null
ComplementNB [0.86842105 0.92982456 0.87719298 0.9122807  0.89380531] 0.8963
DecisionTreeClassifier [0.92105263 0.90350877 0.9122807  0.89473684 0.94690265] 0.9157
DummyClassifier [0.64035088 0.65789474 0.62280702 0.5877193  0.62831858] 0.6274ExtraTreeClassifier [0.94736842 0.96491228 0.90350877 0.89473684 0.94690265] 0.9315
ExtraTreesClassifier [0.96491228 0.98245614 0.96491228 0.95614035 1.        ] 0.9737
GaussianNB [0.93859649 0.96491228 0.9122807  0.93859649 0.95575221] 0.942
GaussianProcessClassifier [0.87719298 0.89473684 0.89473684 0.94736842 0.94690265] 0.9122
GradientBoostingClassifier [0.95614035 0.96491228 0.95614035 0.93859649 0.98230088] 0.9596
HistGradientBoostingClassifier [0.97368421 0.98245614 0.96491228 0.96491228 0.98230088] 0.9737
KNeighborsClassifier [0.92105263 0.92105263 0.92105263 0.92105263 0.95575221] 0.928
LabelPropagation [0.36842105 0.35964912 0.4122807  0.42105263 0.38938053] 0.3902
LabelSpreading [0.36842105 0.35964912 0.4122807  0.42105263 0.38938053] 0.3902
LinearDiscriminantAnalysis [0.94736842 0.98245614 0.94736842 0.95614035 0.97345133] 0.9614
LinearSVC [0.83333333 0.94736842 0.68421053 0.92982456 0.96460177] 0.8719
LogisticRegression [0.93859649 0.95614035 0.88596491 0.95614035 0.96460177] 0.9403
Logisti
'''




