import numpy as np
from sklearn.datasets import load_diabetes, load_boston
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
# 데이터
datasets = load_diabetes()
# datasets = load_boston()

x = datasets.data
y = datasets.target
# print(x.shape, y.shape)
# (442, 10) (442,)
pca = PCA(n_components=9) # 컬럼을 9개로 압축하겠다. 라는뜻 (컬럼을 삭제하겠다는 의미가 아니라, 10개의 컬럼이면 9개로 압축을 하겠다는뜻. 다시 10개로 돌아갈수있다. 대신 손실률 0.00001정도 있다.)
x = pca.fit_transform(x)
# print(x)
# print(x.shape)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
# [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856 0.06027192
#  0.05365605 0.04336832 0.00783199]  # 압축한 것들의 중요도
print(sum(pca_EVR))
# 0.9991439470098977

cumsum = np.cumsum(pca_EVR)
print(cumsum)

# print(np.argmax(cumsum >= 0.94)+1)
# 7

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()
# x_train, x_test, y_train, y_test =  train_test_split(
#     x, y, train_size=0.8, random_state=66, shuffle=True
# )
'''
[[ 0.02793062 -0.09260116  0.02802696 ... -0.01220663 -0.04809855
  -0.00855256]
 [-0.13468605  0.06526341  0.00132778 ... -0.00681271 -0.04818421
   0.01067428]
 [ 0.01294474 -0.07776417  0.0351635  ... -0.05535734 -0.05293076
  -0.02199441]
 ...
 [-0.00976257 -0.05733724  0.02359604 ... -0.00673933 -0.00215418
  -0.03022531]
 [ 0.03295629  0.00999424 -0.04132126 ...  0.00569113 -0.02648904
   0.02595642]
 [-0.09056089  0.18910814 -0.00230125 ...  0.02853071  0.07834495
   0.01166606]]
(442, 9)
'''
# # 모델
# from xgboost import XGBRegressor
# model = XGBRegressor()

# # 훈련
# model.fit(x_train,y_train)

# # 평가, 예측
# results = model.score(x_test, y_test)
# print("결과 : ", results)
# 결과 :  0.999990274544785
# 결과 :  0.9221188601856797

# pca결과
# 결과 :  0.9999349120798557

# 결과 :  0.7994238550951943