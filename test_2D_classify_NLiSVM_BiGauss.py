#%%
import numpy as np
from scipy.linalg import norm
import cvxopt
import cvxopt.solvers
from pylab import *

N = 500        # Data Amount
Slack = 5         # Slack variable
GaussStd = 5.0    # Parameter for Gaussian Kernel


# Gaussian Kernel
def gaussian_kernel(x, y):
    return np.exp(-norm(x-y)**2 / (2 * (GaussStd ** 2)))

# Calculate Descriminalt using only Support Vectors (since a[i] = 0 for non-support vectors)
def f(x, a, t, X, b):
    sum = 0.0
    for n in range(N):
        sum += a[n] * t[n] * gaussian_kernel(x, X[n])
    return sum + b


cls1 = []
cls2 = []

mean1 = [1,  4]
mean2 = [1, -3]
mean3 = [3, -2]
mean4 = [-2, 3]
cov = [[1.0,0.1], [0.5, 0.8]]

cls1.extend(np.random.multivariate_normal(mean1, cov, int(N/4)))
cls1.extend(np.random.multivariate_normal(mean3, cov, int(N/4)))
cls2.extend(np.random.multivariate_normal(mean2, cov, int(N/4)))
cls2.extend(np.random.multivariate_normal(mean4, cov, int(N/4)))

# Create Training Data Matrix X
X = vstack((cls1, cls2))

# Create Lavel t
t = np.array([1.0] * int(N/2) + [-1.0] * int(N/2))


# Calculate Lagrange Multiplier by using Quadratic Programming
K = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        K[i, j] = t[i] * t[j] * gaussian_kernel(X[i], X[j])

Q = cvxopt.matrix(K)
p = cvxopt.matrix(-np.ones(N))
temp1 = np.diag([-1.0]*N)
temp2 = np.identity(N)
G = cvxopt.matrix(np.vstack((temp1, temp2)))
temp1 = np.zeros(N)
temp2 = np.ones(N) * Slack
h = cvxopt.matrix(np.hstack((temp1, temp2)))
A = cvxopt.matrix(t, (1,N))
b = cvxopt.matrix(0.0)
sol = cvxopt.solvers.qp(Q, p, G, h, A, b, kktsolver='ldl', options={'kktreg':1e-9})
a = array(sol['x']).reshape(N)



# Collect Support Vectors (Indexing)
S = []; M = []
for n in range(len(a)):
    if a[n] < 1e-5: continue
    S.append(n)
    if a[n] < Slack:
        M.append(n)


# Calculate bias parameter
sum = 0
for n in M:
    temp = 0
    for m in S:
        temp += a[m] * t[m] * gaussian_kernel(X[n], X[m])
    sum += (t[n] - temp)
b = sum / len(M)



# Show Training Data
plt.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, xlim=(-6, 7), ylim=(-7, 7))
ax.plot(X[t == -1, 0], X[t == -1, 1], "rx")
ax.plot(X[t ==  1, 0], X[t ==  1, 1], "bx")



# Show Discriminant Boundary
X1, X2 = meshgrid(linspace(-6.3,6.3,50), linspace(-6.3,6.3,50))
w, h = X1.shape
X1.resize(X1.size)
X2.resize(X2.size)
Z = array([f(array([i, j]), a, t, X, b) for (i, j) in zip(X1, X2)])
X1.resize((w, h))
X2.resize((w, h))
Z.resize((w, h))
Cown = contour(X1, X2, Z, [0.0], colors='g', linewidths=2, origin='lower')


# -----------------------------------------------------------------------------------------------------------
# Try same thing with sklearn!
from sklearn import svm

clf = svm.SVC(kernel='rbf', C=1/Slack, gamma=GaussStd)
clf.fit(X, t)

X1, X2 = meshgrid(linspace(-6,6,100), linspace(-6,6,100))
ZZ = clf.decision_function(np.c_[X1.ravel(), X2.ravel()])

ZZ.resize(X1.shape)
Cskl = contour(X1, X2, ZZ, [0.0], colors='k', linewidths=2, origin='lower')

h1,_ = Cown.legend_elements()
h2,_ = Cskl.legend_elements()
plt.legend([h1[0], h2[0]], ['this code', 'sklearn'],loc="lower left")

# %%
