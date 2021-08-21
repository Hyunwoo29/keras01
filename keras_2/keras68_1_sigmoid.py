import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x) :
    return 1 / (1 + np.exp(-x)) #exp란 밑이 자연상수 e인 지수함수(e^x)로 변환해줌

x = np.arange(-5, 5, 0.1)
print(len(x))

y = sigmoid(x)

plt.plot(x,y)
plt.grid()
plt.show()