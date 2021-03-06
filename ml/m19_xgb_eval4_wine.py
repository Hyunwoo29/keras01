from xgboost import XGBClassifier,XGBRegressor
from sklearn.datasets import load_boston, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

x,y = load_wine(return_X_y=True)
dataset = load_wine()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size = 0.8, shuffle = True, random_state = 110)


#2. modeling
model = XGBClassifier(n_estimators=10, learning_rate=0.017,n_jobs=8,use_label_encoder=False)

#3. fit
model.fit(x_train,y_train, verbose=1, eval_metric= ['mlogloss','merror','cox-nloglik'],   #다중분류 : mlogloss
            eval_set=[(x_train,y_train),(x_test,y_test)],)
aaa = model.score(x_test,y_test)
print("aaa : ", aaa)


result = model.evals_result()
print(result)