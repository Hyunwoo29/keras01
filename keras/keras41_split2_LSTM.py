#실습
#1~100까지의 데이터를 만드러라
#    x         y
# 1,2,3,4,5    6
#.....
# 95,96,97,98,99,100

import numpy as np
x_data = np.array(range(1, 101))
x_predict = np.array(range(96, 106))
size = 6

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(x_data, size)
predset = split_x(x_predict,size)


# #predict를 만들것
# #96,97,98,99,100 -> 101을 예측하는 모델
# # .....
# #100,101,102,103,104 -> 105
# #예상 predict는 (101, 102,103,104,105)
x = dataset[:, :5]
y = dataset[:, -1]
# print(x.shape, y.shape) # (95, 5) (95,)
x_pred = predset[:, :-1] #(5,5)

# # print("x : \n", x)
# # print("y : ", y)

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2, 
                                            random_state=101)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,train_size=0.8)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred = scaler.transform(x_pred)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1) #(60, 5, 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1) #(19, 5, 1)
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1], 1) #(5, 5, 1)
# print(y_test.shape) # (19,)

# # # # print(x.shape)
# # # # print(x_train.shape)
# # # # print(x_pred.shape)


from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Dense,Input, LSTM

input1 = Input(shape=(5,1))
lstm = LSTM(30, activation='relu') (input1)
dense1 = Dense(40, activation= 'relu') (lstm)
dense1 = Dense(50, activation= 'relu') (dense1)
dense1 = Dense(20, activation= 'relu') (dense1)
output1 = Dense(1) (dense1)

model = Model(input1,output1)

model.compile(loss = 'mse', optimizer= 'adam', metrics = ['mae'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor= 'loss', patience=10, mode='auto')
model.fit(x_train,y_train,epochs=200,batch_size=8,validation_data=(x_val,y_val)
                                , callbacks= early_stopping)

loss = model.evaluate(x_test,y_test)
print(loss)

from sklearn.metrics import r2_score, mean_squared_error

y_pred = model.predict(x_test)
print(y_test.shape)
# print(x_pred1.shape) # (5, 1)
y_pred = x_pred.reshape(x_pred.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
print(y_pred.shape, y_test.shape)
# print(x_pred1)

# r2 = r2_score(y_test, y_pred)
# print('R^2 score : ', r2)

# def RMSE(y_test, y_pred1) :
#     return np.sqrt(mean_squared_error(y_test, x_pred1))

# rmse = RMSE(y_test, x_pred1)
# print("rmse스코어 : ", rmse)