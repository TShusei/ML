#%%

import graphviz
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
import os

os.environ["PATH"] += os.pathsep + 'C:\Program Files\Graphviz/bin/'


wine = load_wine()
X = wine.data
y = wine.target
clf = DecisionTreeClassifier(max_depth = 2)
clf.fit(X, y)


from dtreeviz.trees import dtreeviz # remember to load the package

viz = dtreeviz(clf, X, y,
                target_name="target",
                feature_names=wine.feature_names,
                class_names=list(wine.target_names))

viz
# %%
