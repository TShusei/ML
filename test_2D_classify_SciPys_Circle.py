#%%
import numpy as np
import math
from scipy.linalg import norm
from sklearn.datasets import make_circles

#import cvxopt
#import cvxopt.solvers
import matplotlib.pyplot as plt


X, t = make_circles(n_samples=500, noise=0.2, factor=0.3)
t[t == 0] = -1
t = t.astype('float')
X1, X2 = np.meshgrid(np.linspace(-2,2,200), np.linspace(-2,2,200))


from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF,WhiteKernel, ConstantKernel

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm


models = []

models.append(("SVM-Gauss:",svm.SVC(kernel='rbf', C=1/20, gamma=5.0)))
models.append(("Decision Tree:",DecisionTreeClassifier(max_depth=4)))
models.append(("Random Forest:",RandomForestClassifier(random_state=0)))
models.append(("Neural Net:",MLPClassifier( activation='tanh', solver='lbfgs')))
models.append(("AdaBoost:",AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),n_estimators=300, random_state=0)))
models.append(("GradientBoost:",GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,min_samples_leaf=1, max_depth=1, random_state=0)))
models.append(("GaussianProcess:",gaussian_process.GaussianProcessClassifier(kernel=ConstantKernel() * RBF())))
models.append(("K-Nearest Neighbour:",KNeighborsClassifier(n_neighbors=3)))
models.append(("Gaussian NB:",GaussianNB()))


idx=1
fig = plt.figure(figsize=(12,10))
for name,model in models:
    plt.subplot(3, math.ceil(len(models)/3), idx)
    plt.plot(X[t == -1, 0], X[t == -1, 1], "rx")
    plt.plot(X[t ==  1, 0], X[t ==  1, 1], "bx")

    clf = model.fit(X, t)

    ZZ = clf.predict(np.c_[X1.ravel(), X2.ravel()])
    ZZ = ZZ.reshape(X1.shape)
    plt.contour(X1, X2, ZZ, levels=[0.0], cmap='gray',linewidths=2)
    plt.title(name, fontsize=18)
    idx+=1

plt.subplots_adjust(hspace = 0.3)


# %%
