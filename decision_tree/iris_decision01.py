import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib as mpl

iris = load_iris()
# print(iris)
data = pd.DataFrame(iris.data)
data.columns = iris.feature_names
data["Species"] = iris.target
#print(data)

x = data.iloc[:,:2]
y = data.iloc[:,-1]
#print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=42)

# tree_clf = DecisionTreeClassifier(max_depth=4, criterion='entropy')
# tree_clf.fit(x_train, y_train)
# y_test_hat = tree_clf.predict(x_test)
# print('accuracy score:',accuracy_score(y_test, y_test_hat))
#
# #测试新数据
# print(tree_clf.predict_proba([[5, 1.5]]))
# print(tree_clf.predict([[5, 1.5]]))

#模型超参数自动选择，选取最优深度的模型
depth = np.arange(1, 15)
err_list = []

for i in depth:
    clf = DecisionTreeClassifier(max_depth=i, criterion='entropy')
    clf.fit(x_train, y_train)
    y_test_hat = clf.predict(x_test)
    result = (y_test_hat == y_test)
    # if i == 1:
    #     print(result)
    err = 1 - np.mean(result)
    print(err*100)
    err_list.append(err)
    print(i, '错误率：%.2f%%' % (100 * err))

# 绘图显示
mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(facecolor='w')

plt.plot(depth, err_list, 'ro-', lw=2)
plt.xlabel('决策树深度', fontsize = 15)
plt.ylabel('决策树深度和过拟合', fontsize = 18)
plt.grid(True)
plt.show()
