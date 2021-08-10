# 실습, 모델구성하고 완료
# 회귀 데이터를  Classifier로 만들었을 경우에 에러 확인!

from sklearn.svm import LinearSVC, SVC    # 애네가 먹히는지 확인
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression #  LogisticRegression 면접질문에서 많이나옴 (분류모델이라고 외우면됨)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#실습 diabets
# 1. loss와 r2로 평가를 함
# 2. MinMax와 Standard 결과를 명시

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils import all_estimators

import warnings
warnings.filterwarnings('ignore')
#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

allAlgorithms = all_estimators(type_filter='regressor')

# 데이터 전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train) # train은 훈련을 시키고, test는 훈련에 포함되면 안된다.
x_test = scaler.transform(x_test) 

# print(x.shape, y.shape) # (442, 10) (442,)

# print(datasets.feature_names) # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# print(datasets.DESCR)

# print(y[:30])
# print(np.min(y), np.max(y))
#2. 모델구성


for(name, algorithm) in allAlgorithms :
    try:    
        model = algorithm()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 : ', r2_score(y_test, y_pred))
    except :
        print(name, '은 null')

'''
ARDRegression 의 정답률 :  0.49186844741711055
AdaBoostRegressor 의 정답률 :  0.4210205145469875
BaggingRegressor 의 정답률 :  0.3463944759693899
BayesianRidge 의 정답률 :  0.5011922706777776
CCA 의 정답률 :  0.33899389256622003
DecisionTreeRegressor 의 정답률 :  -0.04866382080160059
DummyRegressor 의 정답률 :  -0.0003678139434499794
ElasticNet 의 정답률 :  0.461052731065413
ElasticNetCV 의 정답률 :  0.49623393177962427
ExtraTreeRegressor 의 정답률 :  -0.12329581553306612
ExtraTreesRegressor 의 정답률 :  0.4150504862043053
GammaRegressor 의 정답률 :  0.4395247238969592
GaussianProcessRegressor 의 정답률 :  -0.8497988098118736
GradientBoostingRegressor 의 정답률 :  0.38567718703109155
HistGradientBoostingRegressor 의 정답률 :  0.3871228855777179
HuberRegressor 의 정답률 :  0.5052980227952182
IsotonicRegression 은 null
KNeighborsRegressor 의 정답률 :  0.3536953490739686
KernelRidge 의 정답률 :  -3.1933147480844664
Lars 의 정답률 :  0.4780144565789868
LarsCV 의 정답률 :  0.4622553218469251
Lasso 의 정답률 :  0.49526707641726697
LassoCV 의 정답률 :  0.49416143669058754
LassoLars 의 정답률 :  0.3702158859753002
LassoLarsCV 의 정답률 :  0.49270413452885686
LassoLarsIC 의 정답률 :  0.5007695981413425
LinearRegression 의 정답률 :  0.5058546730473183
LinearSVR 의 정답률 :  0.30944731161179184
MLPRegressor 의 정답률 :  -0.9164981334377738
MultiOutputRegressor 은 null
MultiTaskElasticNet 은 null
MultiTaskElasticNetCV 은 null
MultiTaskLasso 은 null
MultiTaskLassoCV 은 null
NuSVR 의 정답률 :  0.1378749728050881
OrthogonalMatchingPursuit 의 정답률 :  0.307508795016589
OrthogonalMatchingPursuitCV 의 정답률 :  0.43770301630999153
PLSCanonical 의 정답률 :  -1.3171184615118317
PLSRegression 의 정답률 :  0.48377780813566695
PassiveAggressiveRegressor 의 정답률 :  0.41383996209779994
PoissonRegressor 의 정답률 :  0.5040520476582508
RANSACRegressor 의 정답률 :  0.37904301857272826
RadiusNeighborsRegressor 은 null
RandomForestRegressor 의 정답률 :  0.4329656980816996
RegressorChain 은 null
Ridge 의 정답률 :  0.5053443234962849
RidgeCV 의 정답률 :  0.5025998413243878
SGDRegressor 의 정답률 :  0.5032097140931739
SVR 의 정답률 :  0.14319508768821754
StackingRegressor 은 null
TheilSenRegressor 의 정답률 :  0.49888552197021063
TransformedTargetRegressor 의 정답률 :  0.5058546730473183
TweedieRegressor 의 정답률 :  0.4286578822050584
VotingRegressor 은 null
'''





