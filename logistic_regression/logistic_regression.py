#文件名：logistic_regression.py

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from time import time

iris = datasets.load_iris()
#print(list(iris.keys()))
#print(iris['DESCR'])
#print(iris['feature_names'])

#X = iris['data'][:, 3:]  画图测试用
X = iris['data']
#print(X)
#print(iris['target'])
y = iris['target']
#y = (iris['target'] == 2).astype(np.int)
#print(y)

log_reg = LogisticRegression(multi_class='ovr', solver = 'sag')
log_reg.fit(X, y)

#X_new = np.linspace(0,3,1000).reshape(-1,1)
#print(X_new)

#y_proba = log_reg.predict_proba(X_new)
#y_hat = log_reg.predict(X_new)
#print(y_proba)
#print(y_hat)
print('W', log_reg.coef_)
print('W0', log_reg.intercept_)
#plt.plot(X_new, y_proba[:,1], 'r-', label = 'Iris-Versicolour')
#plt.plot(X_new, y_proba[:,0], 'b-', label = 'Iris-Setosa')
#plt.show()

#print(log_reg.predict([[1.7],[1.5]]))