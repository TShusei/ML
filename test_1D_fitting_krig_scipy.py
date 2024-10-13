


#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF,WhiteKernel, ConstantKernel

#------------------------------------------
# Observed Data
#------------------------------------------

X = np.array([0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.58, 0.61, 0.65, 0.80, 0.85, 0.90])
Y = np.array([0.78, 0.80, 0.76, 0.69, 0.74, 0.71, 0.78, 0.92, 0.80, 0.85, 0.93, 0.98])
pred_X = np.arange(0, 1.1, 0.01)    

X = X.reshape(-1,1)
pred_X = pred_X.reshape(-1,1)


#------------------------------------------
# Ordinary Kriging Estimation
#------------------------------------------

regressor = gaussian_process.GaussianProcessRegressor(kernel=ConstantKernel() * RBF(), normalize_y=True, alpha=0.2)
pred_Y, pred_Std = regressor.fit(X, Y).predict(pred_X, return_std=True)


plt.plot(X, Y, 'ko') 
plt.plot(pred_X, pred_Y, color = 'red')
plt.fill_between(np.arange(0, 1.1, 0.01) , pred_Y - 2 * pred_Std, pred_Y + 2 * pred_Std, alpha=0.2, facecolor='red',)
plt.xlim(0.0, 1.0); plt.ylim(0.6, 1.05)
plt.show()


# %%
