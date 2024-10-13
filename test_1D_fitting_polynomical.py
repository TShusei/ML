#%%
import numpy as np
import matplotlib.pyplot as plt


# Design Matrix ==================================================
def basisFunctionVector (x, coefficient):
    Phi = [x**i  for i in range(coefficient+1)]
    Phi = np.array(Phi).reshape(1, coefficient+1)
    return Phi


## Obs values ------------------------------------------------------------------
X = np.array([0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.58, 0.61, 0.65, 0.80, 0.85, 0.90])
Y = np.array([0.78, 0.80, 0.76, 0.69, 0.74, 0.71, 0.78, 0.92, 0.80, 0.85, 0.93, 0.98])
pred_X = np.arange(0, 1.1, 1.1/100) 

#  Order of coefficient
coefficient = 5

A = np.empty((0,coefficient+1), float)
for i in X:
    A = np.vstack((A, basisFunctionVector (i, coefficient))) 

#A'Ax = A'b, solve (A'A)x=A'b
ATA = np.dot(A.T, A)
Ab = np.dot(A.T, Y)
w = np.linalg.solve(ATA, Ab)

pred_Y = []
for i in pred_X:
    temp = basisFunctionVector (i, coefficient)
    pred_Y.append(np.dot(temp[0],w))

fig = plt.figure(figsize=(8, 6))
plt.plot(X, Y, 'ko') 
plt.plot(pred_X, pred_Y, 'r', label="this code") 
plt.xlim(0.0, 1.0); plt.ylim(0.6, 1.05)




polynp  = np.poly1d(np.polyfit(X, Y, coefficient))

plt.plot(pred_X, polynp(pred_X), 'b--', label="numpy poly1d")
plt.legend(loc="upper left", prop={'size': 18})
plt.xlim(0.0, 1.0); plt.ylim(0.6, 1.05)





# %%
