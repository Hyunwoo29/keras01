import numpy as np
aaa = np.array([1,2,-1000,4,5,6,7,8,90,100,500])

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
    print('1사분위: ',  quartile_1)
    print('q2: ',  q2)
    print('3사분위: ',  quartile_3)
    iqr = quartile_3 - quartile_1   # IQR(InterQuartile Range, 사분범위)
    lower_bound = quartile_1 - (iqr * 1.5)  # 하계
    upper_bound = quartile_3 + (iqr * 1.5)  # 상계

    return np.where((data_out>upper_bound) | (data_out<lower_bound)) 
        
outliers_loc = outliers(aaa)

print('이상치의 위치 :', outliers_loc)

# 시각화
# 위 데이터를 boxplot으로 그리시오!
import matplotlib.pyplot as plt

plt.boxplot(aaa)
plt.show()