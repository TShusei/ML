

#%%
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.interpolate import Rbf


#------------------------------------------
# Observed data generation
numdata = 50

np.random.seed(65537)
x = np.random.random(numdata) 
y = np.random.random(numdata) 

xy = np.ones((numdata,2))
xy[:,0] = x/np.max(x)
xy[:,1] = y/np.max(y)

z  = x**2 + y**2 + 3*x**3 + y + np.random.random(numdata)*4 # Noise

#------------------------------------------
# Mesh to display regression surface
mesh   = 50
xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), mesh), np.linspace(y.min(), y.max(), mesh))



#------------------------------------------
# RBF
#------------------------------------------

rbf = Rbf(x, y, z,  function = 'multiquadric',smooth = 0.01, epsilon = 0.05)
zz = rbf(xx, yy)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, alpha=0.35 , cmap=cm.jet)
plt.title("Radial Basis Function"); 
ax.set_zlim3d(-z.min()-1, z.max()+1)
# %%
