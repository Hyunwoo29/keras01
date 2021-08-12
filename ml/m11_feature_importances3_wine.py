from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = load_wine()
x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.8, random_state=66
)

#2. 모델
# model = DecisionTreeClassifier()
model = DecisionTreeClassifier(max_depth=4)


# 3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)
print("acc : ", acc)


print(model.feature_importances_)
# acc :  0.9444444444444444
# [0.00497399 0.03094929 0.         0.0564906  0.04144514 0.
#  0.1594946  0.         0.         0.         0.         0.33754988
#  0.36909649]

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model) :
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
            align = 'center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1,n_features)

plot_feature_importances_dataset(model)
plt.show()

