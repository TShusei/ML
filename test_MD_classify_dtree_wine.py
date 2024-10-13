#%%

from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn import tree


wine = load_wine()
X = wine.data
y = wine.target
clf = DecisionTreeClassifier(max_depth = 2)
clf.fit(X, y)

fig = plt.figure(figsize=(8,5))
tree.plot_tree(clf)



# %%
