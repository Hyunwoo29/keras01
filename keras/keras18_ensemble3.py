import numpy as np
x1 = np.array([range(100), range(301,  401), range(1, 101)])
# x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
x1 = np.transpose(x1)
# x2 = np.transpose(x2)
# y1 = np.array([range(1001, 1101)]) 
# y1 = np.transpose(y1) -> (100, 1)
y1 = np.array(range(1001, 1101))
y2 = np.array(range(1901, 2001))


from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, y1, y2, test_size=0.2, shuffle=True, random_state=60)

#2.모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1. 앙상블 모델1
input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu', name='dense1')(input1)
dense2 = Dense(7, activation='relu', name='dense2')(dense1)
dense3 = Dense(5, activation='relu', name='dense3')(dense2)


#모델 분기1
output1 = Dense(30, activation= 'relu') (dense1)
output1 = Dense(70)(output1)
output1 = Dense(100)(output1)
output1 = Dense(3)(output1)

#모델 분기2
output2 = Dense(30, activation= 'relu') (dense1)
output2 = Dense(70)(output2)
output2 = Dense(70)(output2)
output2 = Dense(3)(output2)

#2-2. 앙상블 모델2
# input2 = Input(shape=(3,))
# dense11 = Dense(10, activation='relu', name='dense11')(input2)
# dense12 = Dense(10, activation='relu', name='dense12')(dense11)
# dense13 = Dense(10, activation='relu', name='dense13')(dense12)
# dense14 = Dense(10, activation='relu', name='dense14')(dense13)
# output2 = Dense(12, name='output2')(dense14)

# from tensorflow.keras.layers import concatenate, Concatenate
# Concatenate --> 클래스 , concatenate --> 매소드
# 매소드는 클래스 구문에 포함된 함수, 객체의 기능
# merge1 = concatenate([output1, output2])
from tensorflow.keras.layers import Concatenate
# Concatenate = Concatenate()
# merge1 = Concatenate()([output1, output2])
# merge1 = Concatenate(axis=1)([output1, output2])
# merge2 = Dense(10)(merge1)
# merge3 = Dense(5, activation='relu')(merge2)
# last_output = Dense(1)(merge3)

# y1, y2 값이 2개일때
output21 = Dense(7)(output1)
last_output1 = Dense(1, name='last_output1')(output21)

output22 = Dense(8)(output2)
last_output2 = Dense(1, name='last_output2')(output22)

# model = Model(inputs=[input1, input2], outputs=last_output)
model = Model(inputs=(input1), outputs=[last_output1, last_output2])
 
model.summary()

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
model.fit(x1_train, [y1_train, y2_train], epochs=100, batch_size=8, verbose=1)

#4. 평가, 예측
results = model.evaluate(x1_test, [y1_test, y2_test])
print(results)
print("loss : ", results[0])
print("metrics['mae'] : ", results[1])