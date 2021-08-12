from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from xgboost.plotting import plot_importance

# 1. 데이터
# datasets = load_iris()
datasets = load_boston()
x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.8, random_state=66
)

#2. 모델
# model = DecisionTreeClassifier()
model = XGBRegressor()


# 3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)
print("acc : ", acc)
# acc :  0.9666666666666667

print(model.feature_importances_)
# [0.         0.0125026  0.53835801 0.44913938] # 4개 --> iris 컬럼 4개


# [0.03878833 0.         0.         0.         0.00765832 0.29639913
#  0.         0.05991689 0.         0.01862509 0.         0.
#  0.57861225]
import matplotlib.pyplot as plt
import numpy as np

# def plot_feature_importances_dataset(model) :
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_,
#             align = 'center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel("Feature Importances")
#     plt.ylabel("Features")
#     plt.ylim(-1,n_features)

# plot_feature_importances_dataset(model)
# plt.show()

plot_importance(model)
plt.show()