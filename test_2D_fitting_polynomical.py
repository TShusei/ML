

#%%


import itertools
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#-------------------------------------------------------------------
# Function for polynomial 2D

def polyfit2d(x, y, z, order):
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m

#-------------------------------------------------------------------
# Function for polynomial 2D

def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z


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
# Polynomial fitting
#------------------------------------------

m = polyfit2d(x,y,z,3)      # Fit 2d polynomial
zz = polyval2d(xx, yy, m)   # Meshing by polynomial

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, alpha=0.35, cmap=cm.jet)
plt.title("Polynomial Regression"); 
ax.set_zlim3d(-z.min()-1, z.max()+1)
plt.show()
# %%
