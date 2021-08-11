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
from sklearn.model_selection import train_test_split, KFold, cross_val_score
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
kfold = KFold(n_splits=5, shuffle=True, random_state=66)



for(name, algorithm) in allAlgorithms :
    try:    
        model = algorithm()
        scores = cross_val_score(model, x, y, cv=kfold)
        print(name, scores, round(np.mean(scores), 4))
    except :
        print(name, '은 null')


'''
ARDRegression [0.49874835 0.48765748 0.56284846 0.37728801 0.53474369] 0.4923
AdaBoostRegressor [0.36446809 0.47918375 0.52043948 0.39359705 0.45548863] 0.4426
BaggingRegressor [0.34558389 0.41925668 0.34456068 0.35112316 0.37785085] 0.3677
BayesianRidge [0.50082189 0.48431051 0.55459312 0.37600508 0.5307344 ] 0.4893
CCA [0.48696409 0.42605855 0.55244322 0.21708682 0.50764701] 0.438
DecisionTreeRegressor [-0.28542854 -0.28462063 -0.13322083 -0.10275314  0.17082476] -0.127
DummyRegressor [-1.54258856e-04 -2.98519672e-03 -1.53442062e-05 -3.80334913e-03
 -9.58335111e-03] -0.0033
ElasticNet [ 0.00810127  0.00637294  0.00924848  0.0040621  -0.00081988] 0.0054
ElasticNetCV [0.43071558 0.461506   0.49133954 0.35674829 0.4567084 ] 0.4394
ExtraTreeRegressor [-0.00985158 -0.18672125 -0.20862714 -0.23386546 -0.00466665] -0.1287
ExtraTreesRegressor [0.36119026 0.47499209 0.50306641 0.3993558  0.45217223] 0.4382
GammaRegressor [ 0.00523561  0.00367973  0.0060814   0.00174734 -0.00306898] 0.0027
GaussianProcessRegressor [ -5.63607538 -15.27401155  -9.94981401 -12.46884949 -12.04794415] -11.0753
GradientBoostingRegressor [0.3900876  0.48269335 0.48102986 0.39616485 0.44433966] 0.4389
HistGradientBoostingRegressor [0.28899498 0.43812684 0.51713242 0.37267554 0.35643755] 0.3947
HuberRegressor [0.50334705 0.47508237 0.54650576 0.36883712 0.5173073 ] 0.4822
IsotonicRegression [nan nan nan nan nan] nan
KNeighborsRegressor [0.39683913 0.32569788 0.43311217 0.32635899 0.35466969] 0.3673
KernelRidge [-3.38476443 -3.49366182 -4.0996205  -3.39039111 -3.60041537] -3.5938
Lars [ 0.49198665 -0.66475442 -1.04410299 -0.04236657  0.51190679] -0.1495
LarsCV [0.4931481  0.48774421 0.55427158 0.38001456 0.52413596] 0.4879
Lasso [0.34315574 0.35348212 0.38594431 0.31614536 0.3604865 ] 0.3518
LassoCV [0.49799859 0.48389346 0.55926851 0.37740074 0.51636393] 0.487
LassoLars [0.36543887 0.37812653 0.40638095 0.33639271 0.38444891] 0.3742
LassoLarsCV [0.49719648 0.48426377 0.55975856 0.37984022 0.51190679] 0.4866
LassoLarsIC [0.49940515 0.49108789 0.56130589 0.37942384 0.5247894 ] 0.4912
LinearRegression [0.50638911 0.48684632 0.55366898 0.3794262  0.51190679] 0.4876
LinearSVR [-0.33470258 -0.31629592 -0.41902548 -0.3019351  -0.4735802 ] -0.3691
MLPRegressor [-2.83952016 -3.05354364 -3.42686369 -2.77298469 -3.36550909] -3.0917
MultiOutputRegressor 은 null
MultiTaskElasticNet [nan nan nan nan nan] nan
MultiTaskElasticNetCV [nan nan nan nan nan] nan
MultiTaskLasso [nan nan nan nan nan] nan
MultiTaskLassoCV [nan nan nan nan nan] nan
NuSVR [0.14471275 0.17351835 0.18539957 0.13894135 0.1663745 ] 0.1618
OrthogonalMatchingPursuit [0.32934491 0.285747   0.38943221 0.19671679 0.35916077] 0.3121
OrthogonalMatchingPursuitCV [0.47845357 0.48661326 0.55695148 0.37039612 0.53615516] 0.4857
PLSCanonical [-0.97507923 -1.68534502 -0.8821301  -1.33987816 -1.16041996] -1.2086
PLSRegression [0.47661395 0.4762657  0.5388494  0.38191443 0.54717873] 0.4842
PassiveAggressiveRegressor [0.41961663 0.48262552 0.50160841 0.35453582 0.47940456] 0.4476
PoissonRegressor [0.32061441 0.35803358 0.3666005  0.28203414 0.34340626] 0.3341
RANSACRegressor [ 0.38126276  0.33860534  0.20529133 -0.12705812  0.36658568] 0.2329
RadiusNeighborsRegressor [-1.54258856e-04 -2.98519672e-03 -1.53442062e-05 -3.80334913e-03
 -9.58335111e-03] -0.0033
RandomForestRegressor [0.36697006 0.5028351  0.47000005 0.3885205  0.40468814] 0.4266
RegressorChain 은 null
Ridge [0.40936669 0.44788406 0.47057299 0.34467674 0.43339091] 0.4212
RidgeCV [0.49525464 0.48761091 0.55171354 0.3801769  0.52749194] 0.4884
SGDRegressor [0.39331701 0.44189209 0.46451019 0.32963632 0.41497762] 0.4089
SVR [0.14331635 0.18438697 0.17864042 0.1424597  0.1468719 ] 0.1591
StackingRegressor 은 null
TheilSenRegressor [0.50549757 0.47682373 0.54691844 0.34266373 0.53124924] 0.4806
TransformedTargetRegressor [0.50638911 0.48684632 0.55366898 0.3794262  0.51190679] 0.4876
TweedieRegressor [ 0.00585525  0.00425899  0.00702558  0.00183408 -0.00315042] 0.0032
VotingRegressor 은 null
'''

