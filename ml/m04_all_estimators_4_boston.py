# 실습, 모델구성하고 완료
# 회귀 데이터를  Classifier로 만들었을 경우에 에러 확인!

from sklearn.svm import LinearSVC, SVC    # 애네가 먹히는지 확인
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression #  LogisticRegression 면접질문에서 많이나옴 (분류모델이라고 외우면됨)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from sklearn.utils import all_estimators

import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score
datasets = load_boston()
x = datasets.data
y = datasets.target
# print(x.shape)
# print(y.shape)
# print(datasets.feature_names)
# print(datasets.DESCR)
# print(np.min(x), np.max(x))  # 0.0 711.0

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)
allAlgorithms = all_estimators(type_filter='regressor')


#데이터 전처리
# x = x/711.
# x = (x - np.min(x)) / (np.max(x) - np.min(x))

from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer

scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train) # train은 훈련을 시키고, test는 훈련에 포함되면 안된다.
x_test = scaler.transform(x_test)  # 왜냐하면 train은 minmaxcaler가 0~1 이고 test는 0~ 1.2 범위를 넘을 수 있기때문에


for(name, algorithm) in allAlgorithms :
    try:    
        model = algorithm()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 : ', r2_score(y_test, y_pred))
    except :
        print(name, '은 null')

'''
ARDRegression 의 정답률 :  0.8037449551797082
AdaBoostRegressor 의 정답률 :  0.8750736318710777
BaggingRegressor 의 정답률 :  0.8653881826669987
BayesianRidge 의 정답률 :  0.8037638373928007
CCA 의 정답률 :  0.775727268564683
DecisionTreeRegressor 의 정답률 :  0.6875121291129248
DummyRegressor 의 정답률 :  -0.005227869326375867
ElasticNet 의 정답률 :  0.11111593122649766
ElasticNetCV 의 정답률 :  0.7971602685398055
ExtraTreeRegressor 의 정답률 :  0.7393994048079294
ExtraTreesRegressor 의 정답률 :  0.8948667583294889
GammaRegressor 의 정답률 :  0.1552927455615204
GaussianProcessRegressor 의 정답률 :  -2.922037201768416
GradientBoostingRegressor 의 정답률 :  0.9158475207867122
HistGradientBoostingRegressor 의 정답률 :  0.8992462166487565
HuberRegressor 의 정답률 :  0.7673847338137735
IsotonicRegression 은 null
KNeighborsRegressor 의 정답률 :  0.7843808845761827
KernelRidge 의 정답률 :  0.7730664031930281
Lars 의 정답률 :  0.8044888426543626
LarsCV 의 정답률 :  0.8032830033921295
Lasso 의 정답률 :  0.2066075044438942
LassoCV 의 정답률 :  0.8046645285582139
LassoLars 의 정답률 :  -0.005227869326375867
LassoLarsCV 의 정답률 :  0.8044516427844496
LassoLarsIC 의 정답률 :  0.7983441148086403
LinearRegression 의 정답률 :  0.8044888426543627
LinearSVR 의 정답률 :  0.5919736217344909
MLPRegressor 의 정답률 :  0.11598570440353395
MultiOutputRegressor 은 null
MultiTaskElasticNet 은 null
MultiTaskElasticNetCV 은 null
MultiTaskLasso 은 null
MultiTaskLassoCV 은 null
NuSVR 의 정답률 :  0.5641595335839983
OrthogonalMatchingPursuit 의 정답률 :  0.5651272222459415
OrthogonalMatchingPursuitCV 의 정답률 :  0.7415292549226281
PLSCanonical 의 정답률 :  -2.271724502623781
PLSRegression 의 정답률 :  0.7738717095948147
PassiveAggressiveRegressor 의 정답률 :  0.6940060928630944
PoissonRegressor 의 정답률 :  0.5823083831246141
RANSACRegressor 의 정답률 :  -0.42762733995347935
RadiusNeighborsRegressor 의 정답률 :  0.3637807499142598
RandomForestRegressor 의 정답률 :  0.8884361189015941
RegressorChain 은 null
Ridge 의 정답률 :  0.7840559169142114
RidgeCV 의 정답률 :  0.8040739643153296
SGDRegressor 의 정답률 :  0.7724087894667746
SVR 의 정답률 :  0.5591605061048468
StackingRegressor 은 null
TheilSenRegressor 의 정답률 :  0.7596537507435195
TransformedTargetRegressor 의 정답률 :  0.8044888426543627
TweedieRegressor 의 정답률 :  0.1504397995060236
VotingRegressor 은 null
'''

# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()

# model = LinearRegression()
# modle.score :  0.8044888426543627
# model = DecisionTreeRegressor()
# modle.score :  0.6316560051841551
# model = KNeighborsRegressor()
# modle.score :  0.7843808845761827
# model = RandomForestRegressor()
# modle.score :  0.8892545396030381




