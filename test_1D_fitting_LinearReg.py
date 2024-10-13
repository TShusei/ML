#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Design Matrix ==================================================
def basisFunctionVector (x, feature):
    Phi = [x**i  for i in range(feature+1)]
    Phi = np.array(Phi).reshape(1, feature+1)
    return Phi
# ================================================================


## Obs values ------------------------------------------------------------------
X = np.array([0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.58, 0.61, 0.65, 0.80, 0.85, 0.90])
Y = np.array([0.78, 0.80, 0.76, 0.69, 0.74, 0.71, 0.78, 0.92, 0.80, 0.85, 0.93, 0.98])
pred_X = np.arange(0, 1.1, 0.01)    
 
## --------------------------------------------------------------
#  Approach. 1
A = np.array([ X**0, X**1])
w = np.linalg.lstsq(A.T,Y)[0]


pred_Y = w[0] + w[1]*pred_X
plt.plot(X, Y, 'ko')     
plt.plot(pred_X, pred_Y, 'r', label="By solving LSQR")       

## --------------------------------------------------------------
#  Approach. 2
a, b, r_val, p_val, std_err = stats.linregress(X,Y)

pred_Y = a*pred_X+b  
plt.plot(X, Y, 'ko')    
plt.plot(pred_X, pred_Y, 'b--', label="By scipy linregress")   
plt.xlim(0.0, 1.0); plt.ylim(0.6, 1.05)
plt.legend(loc="upper left", prop={'size': 14})
## -------------------------------------------------------------
#  Approach. 3
feature = 1

A = np.empty((0,feature+1), float)
for x in X:
    A = np.vstack((A, basisFunctionVector (x, feature))) 

#A'Ax = A'b, solve (A'A)x=A'b
ATA = np.dot(A.T, A)
Ab = np.dot(A.T, Y)
w = np.linalg.solve(ATA, Ab)

ylist = []
for x in pred_X:
    temp = basisFunctionVector (x, feature)
    ylist.append(np.dot(temp[0],w))

plt.plot(X, Y, 'ko') 
plt.plot(pred_X, ylist, 'r',) 
plt.xlim(0.0, 1.0); plt.ylim(0.6, 1.05)
a=1


# %%
