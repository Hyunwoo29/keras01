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

datasets = load_iris()
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target


# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y[:5])

# print(y.shape) # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  #  LogisticRegression 면접질문에서 많이나옴 (분류모델이라고 외우면됨)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

for(name, algorithm) in allAlgorithms :
    try:    
        model = algorithm()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 : ', r2_score(y_test, y_pred))
    except :
        print(name, '은 null')

'''
ARDRegression 의 정답률 :  0.8989136371853227
AdaBoostRegressor 의 정답률 :  0.8637560119589236
BaggingRegressor 의 정답률 :  0.9261942675159236
BayesianRidge 의 정답률 :  0.9008127082544671
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\cross_decomposition\_pls.py:200: FutureWarning: As of version 0.24, n_components(2) should be in [1, min(n_features, n_samples, n_targets)] = [1, 1]. n_components=1 will be used instead. In version 1.1 (renaming of 0.26), an error will be raised.
  warnings.warn(
CCA 의 정답률 :  0.7858808848970142
DecisionTreeRegressor 의 정답률 :  0.8566878980891719
DummyRegressor 의 정답률 :  -0.0064994150526456185
ElasticNet 의 정답률 :  0.4501970560187023
ElasticNetCV 의 정답률 :  0.9007099660804291
ExtraTreeRegressor 의 정답률 :  0.8925159235668789
ExtraTreesRegressor 의 정답률 :  0.9205656847133757
GammaRegressor 은 없는놈!
GaussianProcessRegressor 의 정답률 :  0.7351597055434437
GradientBoostingRegressor 의 정답률 :  0.8809188458219575
HistGradientBoostingRegressor 의 정답률 :  0.9113000192614582
HuberRegressor 의 정답률 :  0.900771910263751
IsotonicRegression 은 없는놈!
KNeighborsRegressor 의 정답률 :  0.9197452229299363
KernelRidge 의 정답률 :  -0.6618012537064226
Lars 의 정답률 :  0.9011891622636571
LarsCV 의 정답률 :  0.9011891622636571
Lasso 의 정답률 :  -0.0064994150526456185
LassoCV 의 정답률 :  0.9007751097794097
LassoLars 의 정답률 :  -0.0064994150526456185
LassoLarsCV 의 정답률 :  0.9011891622636571
LassoLarsIC 의 정답률 :  0.8890244239239925
LinearRegression 의 정답률 :  0.901189162263657
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\svm\_base.py:985: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  warnings.warn("Liblinear failed to converge, increase "
LinearSVR 의 정답률 :  0.8997634752333424
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:614: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and 
the optimization hasn't converged yet.
  warnings.warn(
MLPRegressor 의 정답률 :  0.8984587645106209
MultiOutputRegressor 은 없는놈!
MultiTaskElasticNet 은 없는놈!
MultiTaskElasticNetCV 은 없는놈!
MultiTaskLasso 은 없는놈!
MultiTaskLassoCV 은 없는놈!
NuSVR 의 정답률 :  0.9238414008048687
OrthogonalMatchingPursuit 의 정답률 :  0.8839165268791885
OrthogonalMatchingPursuitCV 의 정답률 :  0.901189162263657
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\cross_decomposition\_pls.py:200: FutureWarning: As of version 0.24, n_components(2) should be in [1, min(n_features, n_samples, n_targets)] = [1, 1]. n_components=1 will be used instead. In version 1.1 (renaming of 0.26), an error will be raised.
  warnings.warn(
PLSCanonical 의 정답률 :  0.2405000316725262
PLSRegression 의 정답률 :  0.8884091547098087
PassiveAggressiveRegressor 의 정답률 :  0.2714209363974993
PoissonRegressor 의 정답률 :  0.7965590017689269
RANSACRegressor 의 정답률 :  0.901189162263657
RadiusNeighborsRegressor 의 정답률 :  -1.6522760416340166e+17
RandomForestRegressor 의 정답률 :  0.9087746815286624
RegressorChain 은 없는놈!
Ridge 의 정답률 :  0.9003051614269875
RidgeCV 의 정답률 :  0.9003051614269912
SGDRegressor 의 정답률 :  0.8868745559779606
SVR 의 정답률 :  0.9183755111802855
StackingRegressor 은 없는놈!
TheilSenRegressor 의 정답률 :  0.8992056815002801
TransformedTargetRegressor 의 정답률 :  0.901189162263657
TweedieRegressor 의 정답률 :  0.8213325496704202
VotingRegressor 은 없는놈!
'''