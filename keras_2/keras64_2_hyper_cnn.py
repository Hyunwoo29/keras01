# cnn으로 변경
# 파라미터도 변경해봄
# node갯수, activation도 추가
#  에포 [1,2,3]
#러닝레이트 추가
# 나중 과제 : 레이어도 파라미터로 만들어봐라 Dense -> conv
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.backend import batch_dot
from tensorflow.keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000, 28,28,1).astype('float32')/255.

#2. 모델
def build_model(drop=0.5, node = 2, activation = 'relu',optimizer='adam', lr = 0.001):
    inputs = Input(shape=(28,28,1), name = 'input')
    x = Conv2D(filters = 128, kernel_size=(4,4), activation= activation, 
                padding= 'same') (inputs)
    x = Dropout(drop) (x)
    x = Conv2D(filters = 64, kernel_size=(2,2), activation= activation, 
                padding= 'same') (x)
    x = Dropout(drop) (x)
    x = MaxPooling2D(2,2) (x)
    x = Conv2D(filters = 32, kernel_size=(3,3), activation= activation, 
                padding= 'same') (x)
    x = Dropout(drop) (x)
    x = Flatten() (x)
    x = Dense(64/node, activation=activation)(x)
    x = Dropout(drop)(x)
    x = Dense(32/node, activation=activation)(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=lr), metrics=['acc'],
                    loss='categorical_crossentropy')
    return model
def create_hyperparameter():
    optimizers = ['rmsprop', 'adam', 'adadelta']
    activation = ['elu','relu']
    node = [1,2,4]
    validation_split = [0.1,0.2]
    lr = [0.001,0.002]
    return {"activation" : activation, "node" : node, "lr" : lr, 'validation_split' : validation_split, "optimizer" : optimizers}   

hyperparameters = create_hyperparameter()
# print(hyperparameters)
# model2 = build_model()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor = 'val_loss',patience=10)

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2= KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
model = RandomizedSearchCV(model2, hyperparameters, cv=5) # 모델넣고, 하이퍼파라미터 넣고, 크로스 발리데이션 넣음
# 텐서플로우 모델은 랜덤서치에다가 넣을수 없다. 사이킷런 모델 가능.
model.fit(x_train, y_train, verbose=1, batch_size= 312, callbacks = [es], epochs = 10)
 
print(model.best_params_)
print(model.best_score_)

acc = model.score(x_test,y_test)
print("최종 스코어 : ",acc)

# 313/313 [==============================] - 1s 4ms/step - loss: 0.0464 - acc: 0.9872  
# 최종 스코어 :  0.9872000217437744