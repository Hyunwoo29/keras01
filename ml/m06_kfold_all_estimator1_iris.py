import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
from sklearn.metrics import r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, cross_val_score

datasets = load_iris()


x = datasets.data
y = datasets.target




x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  #  LogisticRegression 면접질문에서 많이나옴 (분류모델이라고 외우면됨)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
kfold = KFold(n_splits=5, shuffle=True, random_state=66)
for(name, algorithm) in allAlgorithms :
    try:    
        model = algorithm()

        scores = cross_val_score(model, x, y, cv=kfold)
        print(name, scores, round(np.mean(scores), 4))

    except :
        print(name, '은 null')

'''
AdaBoostClassifier [0.63333333 0.93333333 1.         0.9        0.96666667] 0.8867
BaggingClassifier [0.93333333 0.9        1.         0.9        0.96666667] 0.94BernoulliNB [0.3        0.33333333 0.3        0.23333333 0.3       ] 0.2933    
CalibratedClassifierCV [0.9        0.83333333 1.         0.86666667 0.96666667] 0.9133
CategoricalNB [0.9        0.93333333 0.93333333 0.9        1.        ] 0.9333  
ClassifierChain 은 null
ComplementNB [0.66666667 0.66666667 0.7        0.6        0.7       ] 0.6667   
DecisionTreeClassifier [0.93333333 0.96666667 1.         0.9        0.93333333] 0.9467
DummyClassifier [0.3        0.33333333 0.3        0.23333333 0.3       ] 0.2933ExtraTreeClassifier [0.93333333 0.9        0.96666667 0.93333333 0.96666667] 0.94
ExtraTreesClassifier [0.96666667 0.96666667 1.         0.86666667 0.96666667] 0.9533
GaussianNB [0.96666667 0.9        1.         0.9        0.96666667] 0.9467     
GaussianProcessClassifier [0.96666667 0.96666667 1.         0.9        0.96666667] 0.96
GradientBoostingClassifier [0.93333333 0.96666667 1.         0.93333333 0.96666667] 0.96
HistGradientBoostingClassifier [0.86666667 0.96666667 1.         0.9        0.96666667] 0.94
KNeighborsClassifier [0.96666667 0.96666667 1.         0.9        0.96666667] 0.96
LabelPropagation [0.93333333 1.         1.         0.9        0.96666667] 0.96 
LabelSpreading [0.93333333 1.         1.         0.9        0.96666667] 0.96   
LinearDiscriminantAnalysis [1.  1.  1.  0.9 1. ] 0.98
LinearSVC [0.96666667 0.96666667 1.         0.9        1.        ] 0.9667      
LogisticRegression [1.         0.96666667 1.         0.9        0.96666667] 0.9667
LogisticRegressionCV [1.         0.96666667 1.         0.9        1.        ] 0.9733
MLPClassifier [0.96666667 0.96666667 1.         0.93333333 1.        ] 0.9733  
MultiOutputClassifier 은 null
MultinomialNB [0.96666667 0.93333333 1.         0.93333333 1.        ] 0.9667  
NearestCentroid [0.93333333 0.9        0.96666667 0.9        0.96666667] 0.9333NuSVC [0.96666667 0.96666667 1.         0.93333333 1.        ] 0.9733
OneVsOneClassifier 은 null
OneVsRestClassifier 은 null
OutputCodeClassifier 은 null
PassiveAggressiveClassifier [0.73333333 0.83333333 0.86666667 0.76666667 0.93333333] 0.8267
Perceptron [0.66666667 0.66666667 0.93333333 0.73333333 0.9       ] 0.78       
QuadraticDiscriminantAnalysis [1.         0.96666667 1.         0.93333333 1.  
      ] 0.98
RadiusNeighborsClassifier [0.96666667 0.9        0.96666667 0.93333333 1.      
  ] 0.9533
RandomForestClassifier [1.         0.96666667 1.         0.9        0.96666667] 0.9667
RidgeClassifier [0.86666667 0.8        0.93333333 0.7        0.9       ] 0.84  
RidgeClassifierCV [0.86666667 0.8        0.93333333 0.7        0.9       ] 0.84SGDClassifier [0.93333333 0.8        0.8        0.8        1.        ] 0.8667
SVC [0.96666667 0.96666667 1.         0.93333333 0.96666667] 0.9667
StackingClassifier 은 null
VotingClassifier 은 null
'''