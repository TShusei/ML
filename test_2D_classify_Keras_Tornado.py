#%%

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
#
N = 300 # number of points per class
D = 2 # dimensionality
K = 6 # number of classes
X = np.zeros((N*K,D)) 
y = np.zeros(N*K) 
for j in range(K):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) 
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j


model = Sequential()
model.add(Dense(200, activation='relu', input_dim=D, kernel_initializer='he_uniform'))
model.add(Dense(50, activation='relu', input_dim=D, kernel_initializer='he_uniform'))
model.add(Dense(6, activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
t = to_categorical(y)

#
history = model.fit(X,t, epochs=100)

plt.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.plot(history.epoch, history.history["loss"], label="train_loss")
ax.plot(history.epoch, history.history["accuracy"], label="train_accuracy")
ax.legend()
plt.show()



XX, YY = np.meshgrid(np.linspace(-1.1, 1.1, 200), np.linspace(-1.1, 1.1, 200))
ZZ = model.predict(np.array([XX.ravel(), YY.ravel()]).T)
ZZ = np.argmax(ZZ, axis=1)
ZZ = ZZ.reshape(XX.shape)


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, aspect='equal', xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.gist_rainbow)
ax.contourf(XX, YY, ZZ, cmap=plt.cm.gist_rainbow, alpha=0.2)
ax.contour(XX, YY, ZZ, colors='w', linewidths=0.4)

# %%
