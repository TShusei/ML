#%%
import numpy as np
import cvxopt
import cvxopt.solvers
import matplotlib.pyplot as plt


N = 150  # Data Amount

# Prepare input Data (Training / target) --------------------------------------------------------------------
cls1 = []
cls2 = []

mean1 = [-1, 2]
mean2 = [ 1,-1]
cov = [[1,0.5], [0.5, 1]]

cls1.extend(np.random.multivariate_normal(mean1, cov, int(N/2)))
cls2.extend(np.random.multivariate_normal(mean2, cov, int(N/2)))

X = np.vstack((cls1, cls2))
# -----------------------------------------------------------------------------------------------------------

# Set Target values t ---------------------------------------------------------------------------------------
t = np.array([1.0] * int(N/2) + [-1.0] * int(N/2))

# Compute lagrange multiplier (a) by Quadratic Programming --------------------------------------------------
"""
    cvxopt handles the following QP problem
    min (1/2)aTQa + ITa 
    restrictions: a >= 0 and Ta = 0
"""
K = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        K[i, j] = t[i] * t[j] * np.dot(X[i], X[j])  # Linear Kernel

Q = cvxopt.matrix(K)                        
p = cvxopt.matrix(-np.ones(N))              # N x 1 unit vector 
G = cvxopt.matrix(np.diag([-1.0]*N))        # N x N matrix with diagonal component = -1 
h = cvxopt.matrix(np.zeros(N))              # N x 1 zero vector 
A = cvxopt.matrix(t, (1,N))                 # 1 x N vector with N-target values ?
b = cvxopt.matrix(0.0)                      # Constan 0
sol = cvxopt.solvers.qp(Q, p, G, h, A, b, kktsolver='ldl', options={'kktreg':1e-9})   # Operate Quadratic Programming
a = np.array(sol['x']).reshape(N)              # Get Lagrange Multiplier

# -----------------------------------------------------------------------------------------------------------

# Get Information about Support Vector  ----------------------------------------------------------
# Support vector: a > 0
# Non support vector: a = 0

S = []
for i in range(len(a)):
    if a[i] > 0.001: 
        S.append(i)


# Calculate Weight ---------------------------------------------------------------------------------------
w = np.zeros(2)
for n in S:
    w += a[n] * t[n] * X[n]


# Calculate bias parameter ----------------------------------------------------------------------------------
sum = 0
for n in S:
    temp = 0
    for m in S:
        temp += a[m] * t[m] * np.dot(X[i], X[j])  # Linear Kernel
    sum += (t[n] - temp)
b = sum / len(S)



# Plot Training Data ----------------------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 18})
x1, x2 = np.array(cls1).transpose()
plt.plot(x1, x2, 'rx')

x1, x2 = np.array(cls2).transpose()
plt.plot(x1, x2, 'bx')
# -----------------------------------------------------------------------------------------------------------

# Plot Support Vector ---------------------------------------------------------------------------------------
for n in S:
    plt.scatter(X[n,0], X[n,1], s=80, c='c', marker='o')


# Plot classifier boundary curve ----------------------------------------------------------------------------
x1 = np.linspace(-6, 6, 1000)
yy = - (w[0] / w[1]) * x1 - (b / w[1])   
plt.plot(x1, yy, 'g-', label="SVM - minQP", linewidth=2)




# -----------------------------------------------------------------------------------------------------------
# Try same thing with sklearn!
from sklearn import svm

clf = svm.SVC(kernel='linear')
X_train = np.vstack((cls1, cls2))
t = [-1] * int(N/2) + [1] * int(N/2)

clf.fit(X_train, t)

w = clf.coef_[0]
yy = - (w[0] / w[1]) * x1 - (clf.intercept_[0]) / w[1]

plt.plot(x1,yy,'k--', label="SVM - sklern", linewidth=2)

plt.xlim(-6, 5)
plt.ylim(-5, 6)
plt.legend(loc="upper left")

# %%
