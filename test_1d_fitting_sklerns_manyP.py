
#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Time series data
rng = np.random.RandomState(0)
X = np.sort(rng.uniform(size=100))
Y = np.sin(10 * X) + 5 * X + np.random.normal(0, .3, size=100)
X = X.reshape(-1, 1)
pred_X = np.arange(0, 1.1, 0.01).reshape(-1,1)


# Importing the libraries

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor 
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF,WhiteKernel, ConstantKernel


models = []

models.append(("Ridge Regress:",linear_model.Ridge(alpha=1)))
models.append(("Decision Tree:",DecisionTreeRegressor(max_depth=4)))
models.append(("SVM-gaussian:",SVR(kernel='rbf', C=10, gamma=5,epsilon=0.01)))
models.append(("Random Forest:",RandomForestRegressor(random_state=0)))
models.append(("Neural Net:",MLPRegressor( activation='tanh', solver='lbfgs')))
models.append(("AdaBoost:",AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=300, random_state=0)))
models.append(("GradientBoost:",GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,min_samples_leaf=1, max_depth=1, random_state=0, loss='ls')))
models.append(("GaussianProcess:",gaussian_process.GaussianProcessRegressor(kernel=ConstantKernel() * RBF(), normalize_y=True, alpha=0.2)))
models.append(("K-Nearest Neighbour:",KNeighborsRegressor(n_neighbors=3)))


idx=1
fig = plt.figure(figsize=(12,10))
for name,model in models:
    pred_Y = model.fit(X, Y).predict(pred_X)
    plt.subplot(3, len(models)/3, idx)
    plt.plot(X, Y, 'ko') 
    plt.plot(pred_X, pred_Y, color = 'red')
    plt.xlim(0.0, 1.0); plt.ylim(0.0, 5.5)
    plt.title(name, fontsize=18)
    idx+=1

plt.subplots_adjust(hspace = 0.3)

#results = []
#names = []
#for name,model in models:
#    cv = KFold(n_splits=10, random_state=1, shuffle=True)
#    scores = cross_val_score(model, X, Y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
#    names.append(name)
#    results.append(scores)
#for i in range(len(names)):
 #   print(names[i],results[i].mean()*100)


# %%
