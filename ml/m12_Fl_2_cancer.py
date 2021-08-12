from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

# 1. 데이터
datasets = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.8, random_state=66
)

#2. 모델
# model = DecisionTreeClassifier()
# model = DecisionTreeClassifier(max_depth=4)
# model = GradientBoostingClassifier()
model = XGBClassifier()


# 3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)
print("acc : ", acc)
# acc :  0.9666666666666667

print(model.feature_importances_)
# [0.         0.0125026  0.53835801 0.44913938] # 4개 --> iris 컬럼 4개


# acc :  0.9210526315789473
# [0.         0.0624678  0.         0.         0.         0.
#  0.         0.         0.         0.         0.01297421 0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.01695087 0.         0.75156772 
#  0.         0.         0.00485651 0.146257   0.00492589 0.        ]
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
import pandas as pd
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
cl = df.columns
new_data=[]
for i in range(len(cl)):
    if model.feature_importances_[i] !=0:
       new_data.append(df.iloc[:,i])
new_data = pd.concat(new_data, axis=1)
print(new_data.columns)

new_data1 = new_data.to_numpy()
print(new_data1.shape)

x2_train, x2_test, y2_train, y2_test = train_test_split(new_data1, datasets.target, train_size=0.8, random_state=32)
model2 = DecisionTreeClassifier(max_depth=4)   
model2.fit(x2_train,y2_train)
acc2 = model2.score(x2_test, y2_test)
print("acc :", acc2)


import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model) :
    n_features = new_data1.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
            align = 'center')
    plt.yticks(np.arange(n_features), new_data.columns)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1,n_features)

plot_feature_importances_dataset(model2)
plt.show()
