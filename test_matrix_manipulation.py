#%%

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

#------------------------------------------
# Test Matrix operation for Python
A = np.matrix( [[-1,2,3],[11,-12,13],[21,22,-23]])
x = np.matrix( [[1],[2],[3]] )
y = np.matrix( [[1,2,3]] )

#------------------------------------------
# Matrix Manipulation
TrnA = A.T
MltA = A*x
InvA = A.I
DotA = np.dot(A,x)
SlvA = np.linalg.solve(A, x)
DetA = np.linalg.det(A)
SvdA = np.linalg.svd(A)

im = plt.imshow(TrnA, cmap=cm.RdYlGn, vmax=abs(TrnA).max(), vmin=-abs(TrnA).max())
cbar = plt.colorbar(im)

plt.rcParams.update({'font.size': 16})
# %%
