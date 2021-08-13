# dnn, cnn과 비교!
# RandomSearch 로도 해볼것
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import train_test_split as ts
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
import datetime

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x = np.append(x_train,x_test,axis=0)

print(x_train.shape)  # 60000,28,28
print(x_test.shape)   # 10000, 28,28

x_train = x_train.reshape(x_train.shape[0],28*28)/255
x_test = x_test.reshape(x_test.shape[0],28*28)/255


# pca = PCA()
# pca.fit(x_train)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# # print(cumsum)   
# '''
# d : 154
# '''

# d = np.argmax(cumsum >= 1.0)+1
# print("cumsum >= 0.95", cumsum>= 1.0)
# print("d :", d)

pca = PCA(n_components=154)
x_train = pca.fit_transform(x_train)  # merge fit,transform
x_test = pca.transform(x_test)


parameters = [
    {"n_estimators":[100,200,300], "learning_rate":[0.1,0.3, 0.001, 0.01],
     "max_depth":[4,5,6]},
     {"n_estimators":[90,100,110], "learning_rate":[0.1,0.001,0.01],
     "max_depth":[4,5,6],"colsample_bytree":[0.6,0.9,1] },
    {"n_estimators":[90, 110], "learning_rate":[0.1, 0.001, 0.5]
    ,"max_depth":[4,5,6],"colsample_bytree":[0.6,0.9,1],
     "colsample_bylevel":[0.6,0.7,0.9] }
]

models = [GridSearchCV,RandomizedSearchCV]
for i in models:
    model = i
    print("\n",f'{i.__name__}')     
    model = i(XGBClassifier(n_jobs=8,use_label_encoder=False) , parameters, cv=5)
    start = datetime.datetime.now()
    model.fit(x_train, y_train,eval_metric='mlogloss',verbose = True, 
                eval_set=[(x_train,y_train),(x_test,y_test)])
    end = datetime.datetime.now()
    print("걸린시간 : ",(start-end))            
    results = model.score(x_test,y_test)
    print("best parameter : ", model.best_estimator_)
    print('acc : ',results)
