import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


aaa = np.array([[1,2,10000,3,4,6,7,8,90,100,5000],
               [1000,2000,3,4000,5000,6000,7000,8,9000,10000,1001]])
aaa = aaa.transpose()
print(aaa.shape) # (10, 2)

from sklearn.covariance import EllipticEnvelope

outliers = EllipticEnvelope(contamination=.2)
outliers.fit(aaa)

results = outliers.predict(aaa)

print(results)