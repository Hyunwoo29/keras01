import numpy as np
import pandas as pd
from scipy import stats


num = 4   # 파일의 갯수

x = []
for i in range(1,num+1):
    df = pd.read_csv(f'./_data/pattern_mlp{i}.csv', index_col=0, header=0)
    data = df.to_numpy()
    x.append(data)

x = np.array(x)

# print(x.shape)
a= []
df = pd.read_csv(f'./_data/pattern_mlp{i}.csv', index_col=0, header=0)
for i in range(602):
    for j in range(1):
        b = []
        for k in range(num):
            b.append(x[k,i,j].astype('int'))
        a.append(stats.mode(b)[0]) 
# a = np.array(a)
# a = a.reshape(72000,4)

# print(a)

sub = pd.read_csv('./_data/sample_submission.csv')
sub['label'] = np.array(a)
sub.to_csv('./_data/bbb1.csv',index=False)