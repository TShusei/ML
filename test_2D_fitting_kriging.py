
#%%


import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def Test_Okriging(x,y,Obsp,xx,yy,nget,sill,rnge):

    nobs = len(Obsp)

    A  = np.ones((nobs+1,nobs+1)) # = 1 for lagrange multiplier
    b  = np.ones((nobs+1,1))
    Cd = np.ones((nobs+1,nobs+1))

    b[nobs]= 1   # 1 = lagrange multiple

    pos = np.ones((nobs,2))
    pos[:,0] = x/np.max(x)
    pos[:,1] = y/np.max(y)

    # Variogram_Generation(Cd,A,Obsp,pos,nget,sill,rnge)
    #------------------------------------------------
    #  Covariance Generation by Data-Data Distance
    for i in range(nobs):
        for j in range(i, nobs) :
            Cd[i][j] = np.linalg.norm(pos[i]-pos[j])
    #------------------------------------------------


    #------------------------------------------------
    # Variogram: Spherical Method
    for i in range(nobs) :
        for j in range(i, nobs) :
            A[i][j] = A[j][i] = nget + sill*(1.5*Cd[i][j]/rnge - 0.5*(Cd[i][j]/rnge)**3)
            A[i][j] = A[j][i] = nget + sill*np.exp(-3*Cd[i][j]**2/(rnge**2))
    #------------------------------------------------


    #---------initialize values--------
    posidim = len(pos[0])
    mesh = len(xx)

    cnt_x = (1. - 0.)/mesh
    cnt_y = (1. - 0.)/mesh

    vvval = np.ones((int(mesh),int(mesh)))


    #--------- estimate all location --------
    cord  = np.ones((posidim))

    for i in range(mesh):
        for j in range(mesh):
            cord[0] = xx[i,j]
            cord[1] = yy[i,j]

            #---Current to Obs Distance Yo ------------------
            for k in range(nobs) :

                distance = np.linalg.norm(cord-pos[k])
                b[k] = nget + sill*(1.5*distance/rnge - 0.5*(distance/rnge)**3)
                b[k] = nget + sill*np.exp(-3*distance**2/(rnge**2))

            Weit  = np.linalg.solve(A,b)

            OKest = np.sum([Weit[i]*Obsp[i] for i in range(0, nobs)])

            vvval[i,j] = OKest

    #--------- Return! ---------
    return vvval




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
# Ordinary Kriging Estimation
#------------------------------------------

zz = Test_Okriging(x,y,z,xx,yy,0.1,0.1,0.15)

fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
ax.plot_surface(xx, yy, zz,rstride = 1, cstride = 1, alpha=0.35, cmap=cm.jet)
plt.title("Ordinary Kriging"); 
ax.set_zlim3d(-z.min()-1, z.max()+1)



# %%
