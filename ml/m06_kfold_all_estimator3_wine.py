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
from sklearn.model_selection import train_test_split, KFold, cross_val_score
datasets = load_wine()


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
AdaBoostClassifier [0.88888889 0.86111111 0.88888889 0.94285714 0.97142857] 0.9106
BaggingClassifier [1.         0.91666667 0.88888889 0.97142857 1.        ] 0.9554
BernoulliNB [0.41666667 0.47222222 0.27777778 0.48571429 0.34285714] 0.399
CalibratedClassifierCV [0.94444444 0.94444444 0.88888889 0.88571429 0.91428571] 0.9156
CategoricalNB [       nan        nan        nan 0.94285714        nan] nan
ClassifierChain 은 null
ComplementNB [0.69444444 0.80555556 0.55555556 0.6        0.6       ] 0.6511
DecisionTreeClassifier [0.91666667 0.97222222 0.91666667 0.82857143 0.94285714] 0.9154
DummyClassifier [0.41666667 0.47222222 0.27777778 0.48571429 0.34285714] 0.399
ExtraTreeClassifier [0.83333333 0.91666667 0.83333333 0.85714286 0.88571429] 0.8652
ExtraTreesClassifier [1.         0.97222222 1.         0.97142857 1.        ] 0.9887
GaussianNB [1.         0.91666667 0.97222222 0.97142857 1.        ] 0.9721
GaussianProcessClassifier [0.44444444 0.30555556 0.55555556 0.62857143 0.45714286] 0.4783
GradientBoostingClassifier [0.97222222 0.91666667 0.88888889 0.97142857 0.97142857] 0.9441
HistGradientBoostingClassifier [0.97222222 0.94444444 1.         0.97142857 1.        ] 0.9776
KNeighborsClassifier [0.69444444 0.77777778 0.61111111 0.62857143 0.74285714] 0.691
LabelPropagation [0.52777778 0.47222222 0.5        0.4        0.54285714] 0.4886
LabelSpreading [0.52777778 0.47222222 0.5        0.4        0.54285714] 0.4886
LinearDiscriminantAnalysis [1.         0.97222222 1.         0.97142857 1.        ] 0.9887
LinearSVC [0.80555556 0.86111111 0.52777778 0.8        0.85714286] 0.7703
LogisticRegression [0.97222222 0.94444444 0.94444444 0.94285714 1.        ] 0.9608
LogisticRegressionCV [1.         0.94444444 0.97222222 0.94285714 0.97142857] 0.9662
MLPClassifier [0.38888889 0.91666667 0.72222222 0.65714286 0.6       ] 0.657
MultiOutputClassifier 은 null
MultinomialNB [0.77777778 0.91666667 0.86111111 0.82857143 0.82857143] 0.8425
NearestCentroid [0.69444444 0.72222222 0.69444444 0.77142857 0.74285714] 0.7251
NuSVC [0.91666667 0.86111111 0.91666667 0.85714286 0.8       ] 0.8703
OneVsOneClassifier 은 null
OneVsRestClassifier 은 null
OutputCodeClassifier 은 null
PassiveAggressiveClassifier [0.25       0.27777778 0.30555556 0.6        0.6       ] 0.4067
Perceptron [0.61111111 0.80555556 0.47222222 0.48571429 0.62857143] 0.6006
QuadraticDiscriminantAnalysis [0.97222222 1.         1.         1.         1.        ] 0.9944
RadiusNeighborsClassifier [nan nan nan nan nan] nan
RandomForestClassifier [1.         0.94444444 1.         0.97142857 1.        ] 0.9832
RidgeClassifier [1.         1.         1.         0.97142857 1.        ] 0.9943
RidgeClassifierCV [1.         1.         1.         0.97142857 1.        ] 0.9943
SGDClassifier [0.63888889 0.5        0.55555556 0.65714286 0.51428571] 0.5732
SVC [0.69444444 0.69444444 0.61111111 0.62857143 0.6       ] 0.6457
StackingClassifier 은 null
VotingClassifier 은 null
'''