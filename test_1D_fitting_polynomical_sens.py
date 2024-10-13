#%%

import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------
# Get polynomial Coefficient : 1D, Random
X = np.array([0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.58, 0.61, 0.65, 0.80, 0.85, 0.90])
Y = np.array([0.78, 0.80, 0.76, 0.69, 0.74, 0.71, 0.78, 0.92, 0.80, 0.85, 0.93, 0.98])
pred_X = np.arange(0, 1.1, 0.01)    


# How much degree ?
p3  = np.poly1d(np.polyfit(X, Y, 2))
p6  = np.poly1d(np.polyfit(X, Y, 4))
p10 = np.poly1d(np.polyfit(X, Y, 8))
p30 = np.poly1d(np.polyfit(X, Y, 10))

# Shou picture
fig = plt.figure(figsize=(8, 6))

plt.plot(pred_X, p3(pred_X), '-', label="2nd")
plt.plot(pred_X, p6(pred_X), '-', label="4th")
plt.plot(pred_X, p10(pred_X), '-', label="8th")
plt.plot(pred_X, p30(pred_X), '-', label="12th")
plt.plot(X, Y, '.k',  markersize=13,linewidth = 2)
plt.legend(loc="upper left")

plt.xlim(0.0, 1.0); plt.ylim(0.6, 1.05)
# %%
