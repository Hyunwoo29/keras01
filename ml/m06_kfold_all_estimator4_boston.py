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
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score
import numpy as np
from sklearn.utils import all_estimators

import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score
datasets = load_boston()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)
allAlgorithms = all_estimators(type_filter='regressor')
kfold = KFold(n_splits=5, shuffle=True, random_state=66)




for(name, algorithm) in allAlgorithms :
    try:    
        model = algorithm()
        scores = cross_val_score(model, x, y, cv=kfold)
        print(name, scores, round(np.mean(scores), 4))
    except :
        print(name, '은 null')


'''
ARDRegression [0.80125693 0.76317071 0.56809285 0.6400258  0.71991866] 0.6985
AdaBoostRegressor [0.90230011 0.79170453 0.75843279 0.82977895 0.85832961] 0.8281
BaggingRegressor [0.89592196 0.82564896 0.76410012 0.86409736 0.89280842] 0.8485
BayesianRidge [0.79379186 0.81123808 0.57943979 0.62721388 0.70719051] 0.7038
CCA [0.79134772 0.73828469 0.39419624 0.5795108  0.73224276] 0.6471
DecisionTreeRegressor [0.702314   0.6768117  0.8217509  0.71678541 0.83473357] 0.7505
DummyRegressor [-0.00053702 -0.03356375 -0.00476023 -0.02593069 -0.00275911] -0.0135
ElasticNet [0.73383355 0.76745241 0.59979782 0.60616114 0.64658354] 0.6708
ElasticNetCV [0.71677604 0.75276545 0.59116613 0.59289916 0.62888608] 0.6565
ExtraTreeRegressor [0.74054771 0.65352028 0.44399133 0.70516059 0.69636082] 0.6479
ExtraTreesRegressor [0.93221463 0.87071389 0.76839729 0.88669289 0.93796581] 0.8792
GammaRegressor [-0.00058757 -0.03146716 -0.00463664 -0.02807276 -0.00298635] -0.0136
GaussianProcessRegressor [-6.07310526 -5.51957093 -6.33482574 -6.36383476 -5.35160828] -5.9286
GradientBoostingRegressor [0.94653735 0.83446222 0.82703014 0.88540234 0.93055257] 0.8848
HistGradientBoostingRegressor [0.93235978 0.82415907 0.78740524 0.88879806 0.85766226] 0.8581
HuberRegressor [0.74400323 0.64244715 0.52848946 0.37100122 0.63403398] 0.584
IsotonicRegression [nan nan nan nan nan] nan
KNeighborsRegressor [0.59008727 0.68112533 0.55680192 0.4032667  0.41180856] 0.5286
KernelRidge [0.83333255 0.76712443 0.5304997  0.5836223  0.71226555] 0.6854
Lars [0.77467361 0.79839316 0.5903683  0.64083802 0.68439384] 0.6977
LarsCV [0.80141197 0.77573678 0.57807429 0.60068407 0.70833854] 0.6928
Lasso [0.7240751  0.76027388 0.60141929 0.60458689 0.63793473] 0.6657
LassoCV [0.71314939 0.79141061 0.60734295 0.61617714 0.66137127] 0.6779
LassoLars [-0.00053702 -0.03356375 -0.00476023 -0.02593069 -0.00275911] -0.0135
LassoLarsCV [0.80301044 0.77573678 0.57807429 0.60068407 0.72486787] 0.6965
LassoLarsIC [0.81314239 0.79765276 0.59012698 0.63974189 0.72415009] 0.713
LinearRegression [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215] 0.7128
LinearSVR [ 0.59137815  0.71889265  0.26280517  0.54354659 -1.98845728] 0.0256
MLPRegressor [0.55026612 0.69921415 0.46008908 0.42141371 0.5104602 ] 0.5283
MultiOutputRegressor 은 null
MultiTaskElasticNet [nan nan nan nan nan] nan
MultiTaskElasticNetCV [nan nan nan nan nan] nan
MultiTaskLasso [nan nan nan nan nan] nan
MultiTaskLassoCV [nan nan nan nan nan] nan
NuSVR [0.2594254  0.33427351 0.263857   0.11914968 0.170599  ] 0.2295
OrthogonalMatchingPursuit [0.58276176 0.565867   0.48689774 0.51545117 0.52049576] 0.5343
OrthogonalMatchingPursuitCV [0.75264599 0.75091171 0.52333619 0.59442374 0.66783377] 0.6578
PLSCanonical [-2.23170797 -2.33245351 -2.89155602 -2.14746527 -1.44488868] -2.2096
PLSRegression [0.80273131 0.76619347 0.52249555 0.59721829 0.73503313] 0.6847
PassiveAggressiveRegressor [-0.06846872  0.24698043 -0.57083478 -0.00800053  0.36520655] -0.007
PoissonRegressor [0.85659255 0.8189989  0.66691488 0.67998192 0.75195656] 0.7549
RANSACRegressor [ 0.76424028 -0.09028351  0.57223797  0.37562769  0.66507041] 0.4574
RadiusNeighborsRegressor [nan nan nan nan nan] nan
RandomForestRegressor [0.92530646 0.85526788 0.8169916  0.88684263 0.89930926] 0.8767
RegressorChain 은 null
Ridge [0.80984876 0.80618063 0.58111378 0.63459427 0.72264776] 0.7109
RidgeCV [0.81125292 0.80010535 0.58888304 0.64008984 0.72362912] 0.7128
SGDRegressor [-1.18417888e+26 -1.03435608e+26 -9.48704116e+26 -1.65871817e+26
 -3.33423013e+26] -3.3397048849282805e+26
SVR [0.23475113 0.31583258 0.24121157 0.04946335 0.14020554] 0.1963
StackingRegressor 은 null
TheilSenRegressor [0.79605755 0.71572179 0.58902721 0.55319339 0.70985208] 0.6728
TransformedTargetRegressor [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215] 0.7128
TweedieRegressor [0.7492543  0.75457294 0.56286929 0.57989884 0.63242475] 0.6558
VotingRegressor 은 null
'''


