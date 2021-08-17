# coefficient 계수

x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
y = [-3, 65, -19, 11, 3, 47, -1, -7, -47, -25]

import pandas as pd
df = pd.DataFrame({'X':x, 'Y':y})


x_train = df.loc[:,'X']    # 10,
y_train = df.loc[:,'Y']    # 10,

print(x_train.shape, y_train.shape)

x_train = x_train.values.reshape(len(x_train),1) 

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)


score = model.score(x_train, y_train)
print("score: ", score) 

print("기울기(weight) : ",model.coef_)  
print("절편(bias) : ",model.intercept_) 