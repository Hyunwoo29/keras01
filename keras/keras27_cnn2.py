from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential()                                         
model.add(Conv2D(10, kernel_size=(2,2), #(N, 10, 10, 1)
                    padding='same', input_shape=(10, 10, 1))) #(N, 10, 10, 10)
model.add(Conv2D(20, (2,2), activation='relu')) #(N, 9, 9, 20)
model.add(Conv2D(30, (2,2), padding='valid'))  # (N, 8, 8, 30)
model.add(MaxPooling2D())   #(N, 4, 4, 30)
model.add(Conv2D(15, (2,2)))
model.add(Flatten())     # (N, 180)                 
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# 과제 1. Conv20의 디폴트 엑티베이션
# 과제 2. summary의 파라미터 갯수 완벽 이해
 