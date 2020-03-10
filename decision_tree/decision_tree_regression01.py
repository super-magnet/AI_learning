import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

N = 100
X = np.random.rand(N)*6 - 3
X.sort()

y = np.sin(X) + np.random.rand(N) * 0.05
#print(y)
#print(X)
X = X.reshape(-1,1)  #将一维数组转化为矩阵
#print(X)

dt_reg = DecisionTreeRegressor(criterion='mse', max_depth=3)
dt_reg.fit(X, y)

X_test = np.linspace(-3,3,50).reshape(-1,1)
y_hat = dt_reg.predict(X_test)

plt.plot(X, y, 'y*', label='actual')
plt.plot(X_test, y_hat, 'b-', linewidth=2, label='predict')
plt.legend(loc = 'upper left')
plt.grid()
plt.show()

# 采用for循环，比较不同深度的决策树的拟合、预测效果
depth = [2, 4, 6, 8, 10]
colors = 'rgbmy'
dt_reg_0 = DecisionTreeRegressor()
plt.plot(X, y, 'ko', label='actual')
for d, c in zip(depth, colors):   #for循环中使用zip()，同时并列迭代两个参数
    dt_reg_0.set_params(max_depth=d)
    dt_reg_0.fit(X, y)
    y_hat_0 = dt_reg_0.predict(X_test)
    plt.plot(X_test, y_hat_0, '-', color=c, linewidth=2, label="depth=%d"%d)
plt.legend(loc='upper left')
plt.grid(b=True)
plt.show()
