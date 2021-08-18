import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


aaa = np.array([[1,2,10000,3,4,6,7,8,90,100,5000],
               [1000,2000,3,4000,5000,6000,7000,8,9000,10000,1001]])
aaa = aaa.transpose()
print(aaa.shape) # (10, 2)


def outliers(data_out):
    allout = []
    for i in range(data_out.shape[1]):
        quartile_1, q2, quartile_3 = np.percentile(data_out[:,i], [25, 50, 75])
        print('1사분위(25%지점): ',  quartile_1)
        print('q2(50%지점): ',  q2)
        print('3사분위(75%지점): ',  quartile_3)
        iqr = quartile_3 - quartile_1   # IQR(InterQuartile Range, 사분범위)
        print('iqr: ', iqr)
        lower_bound = quartile_1 - (iqr * 1.5)  # 하계
        upper_bound = quartile_3 + (iqr * 1.5)  # 상계
        print('lower_bound: ', lower_bound)
        print('upper_bound: ', upper_bound)

        a = np.where((data_out[:,i]>upper_bound) | (data_out[:,i]<lower_bound)) 
        allout.append(a)

    return np.array(allout)


outlier_loc = outliers(aaa)
print('이상치의 위치: ', outlier_loc)