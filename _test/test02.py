from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape)
print(y.shape)
print(datasets.feature_names)
print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=60)

model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

loss = model.evaluate(x_test, y_test)
print('lose : ', loss)
y_predict = model.predict(x_test)
print('예측: ', y_predict)
r2 = r2_score(y_test, y_predict)
print('r2socre: ', r2)

# lose :  29.619949340820312
# r2socre:  0.7030551173717354

# lose :  23.3414306640625
# r2socre:  0.7174745925787925

# lose :  22.402912139892578
# r2socre:  0.7288344441347756