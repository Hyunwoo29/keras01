import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Conv1D, Conv2D, Flatten, MaxPooling1D, MaxPooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D, Dropout, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.utils.np_utils import normalize
# 실습 1
# men women 데이터로 모델링 구성할것!
import numpy as np
x_train = np.load('./_save/_npy/k59_3_train_x.npy')
y_train = np.load('./_save/_npy/k59_3_train_y.npy')
x_test = np.load('./_save/_npy/k59_3_test_x.npy')
y_test = np.load('./_save/_npy/k59_3_test_y.npy')
print(x_train.shape) # (200, 300, 300, 3)
print(y_train.shape) # (200,)
print(x_test.shape)  # (200, 300, 300, 3)
print(y_test.shape)  # (200,)

# 2. 모델
model=Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(200,200,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(2,activation="softmax"))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=128,epochs=30)

model.save('./_save/men_women.h5')
loss = model.evaluate(x_train, y_test)



# 위에거로 시각화 할것

print('acc: ', loss[0])
print('val_acc: ', loss[1])
# 실습2, 과제: 내가 남자인지 여자인지? acc몇입니까?
# 본인 사진으로 predict 하시오!

from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
from PIL import Image
import numpy as np
import os
import cv2

def convert_to_array(img):
    im = cv2.imread(img)
    cv_rgb =cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    plt.imshow(cv_rgb)
    plt.show()
    img_ = Image.fromarray(im, 'RGB')
    image = img_.resize((200, 200))
    return np.array(image)

def get_cell_name(label):
    if label==0:
        return "men"
    if label==1:
        return "women"
    
def predict_cell(file):
    model = load_model('./_save/men_women.h5')
    print("Predicting Type of people Image.................................")
    ar=convert_to_array(file)
    ar=ar/255
    label=1
    a=[]
    a.append(ar)
    a=np.array(a)
    score=model.predict(a,verbose=1)
    print(score)
    label_index=np.argmax(score)
    print(label_index)
    acc=np.max(score)
    Cell=get_cell_name(label_index)
    return Cell,"The people Cell is a "+Cell+" with accuracy =    "+str(acc)


predict_cell('./_data/predict/00002341.jpg')
