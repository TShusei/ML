
#%%
#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

# == Model selection by AIC ===================================
# In this sample, linear regression model is tested.
# True value is approximated by:
#   true = y + epsilon

# The gereral algotithm is:
# 1) Determine the features used for AIC estimation.
# 2) Construct log-likelihood function.
# 3) Calculate maximum log-likelihood estimator.
# 4) Calculate AIC
# 5) Go back to #1 and change the features
# ============================================================


# Design Matrix ==============================================
def basisFunctionVector (x, feature):
    Phi = [x**i  for i in range(feature+1)]
    Phi = np.array(Phi).reshape(1, feature+1)
    return Phi
# ============================================================



## Set Parameters for AIC -------------------------------------
MaxAIC = 20          # maximum polynomial
LL   = []            # log-likelihood
AIC  = []            # Calculated AIC          


fig = plt.figure(figsize=(15,10))
for feature in range(MaxAIC):

    ## Set Input parameters ----------------------------------------------------
    N = 20                              # number of data
    X = np.linspace(0.05, 0.95, N)         # x parameter
    Y = np.array([0.85, 0.78, 0.70, 0.76, 0.77, 0.69, 0.80, 0.74, 0.76, 0.76,
                  0.81, 0.79, 0.80, 0.84, 0.88, 0.89, 0.86, 0.93, 0.97, 0.98])
    pred_X = np.arange(0, 1, 0.01)
    ## -------------------------------------------------------------------------


    A = np.empty((0,feature+1), float)
    for x in X:
        A = np.vstack((A, basisFunctionVector (x, feature))) 

    ## Compute weight (Maximum likelihood) ------------------------------------
    w = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, Y))


    ## Compute SigmaSq of Gaussian Noise (Maximum likelihood) -----------------
    SigmaSq = 0
    for i in range(len(X)):
        SigmaSq += (Y[i] - np.dot(A[i],w))**2
    SigmaSq = SigmaSq / len(X)


    ## Compute log-likelihood / AIC (Gaussian Noise Case) ----------------------
    LL.append(-0.5*N*np.log(2*np.pi*SigmaSq)-N/2)       
    AIC.append(-2*LL[feature]+2*(len(w)+1))

    plt.subplot(4, MaxAIC/4, feature+1)

    plt.xlim(0.0, 1.0)
    plt.ylim(0.6, 1.05)
    plt.plot(X, Y, 'ko')    # Obs data   
    pred_Y = []
    for x in pred_X:
        temp = basisFunctionVector (x, feature)
        pred_Y.append(np.dot(temp[0],w))
    plt.plot(pred_X, pred_Y, 'r')   
    plt.title("Polynomial order = " + str(feature)); 

plt.subplots_adjust(hspace = 0.3)
plt.show()

fig = plt.figure(figsize=(11,5))
plt.subplot(1, 2, 1)
plt.rcParams.update({'font.size': 18})
xlist = np.arange(1, MaxAIC+1, 1)
plt.plot(xlist, LL, 'bo-')
plt.xlabel('Order of Polynomical')
plt.title("Log-likelihood"); 

plt.subplot(1, 2, 2)
plt.rcParams.update({'font.size': 18})
plt.plot(xlist, AIC, 'bo-')
plt.xlabel('Order of Polynomical')
plt.title("AIC"); 
plt.show()

# %%
