from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

iris = load_iris()
x = iris.data[:, :2] #花萼长度和宽度
print(x)
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=20, max_leaf_nodes=16, n_jobs=5)
#树太多时（n_estimators值过大时），也会产生过拟合现象
rnd_clf.fit(x_train, y_train)

#bagging并行思想，此处创建Bagging树群，和随机森林类直接创建森林，本质一样
bag_clf = BaggingClassifier(DecisionTreeClassifier(splitter='random', max_leaf_nodes=16),
                             n_estimators=20, max_samples=1.0, bootstrap=True, n_jobs=5)
y_test_pred = rnd_clf.predict(x_test)
print('acc score of RandomForestClassifier:', accuracy_score(y_test_pred, y_test))

bag_clf.fit(x_train, y_train)
y_test_bag = bag_clf.predict(x_test)
print('acc score of BaggingClassifier:', accuracy_score(y_test_bag, y_test))

#Feature_Importance求相关性
#常用求相关性方法（1）pearson相关系数（线性相关性）（2）L1正则，Lasso_Regression （3)树方法求重要特征
iris = load_iris()
rnd_clf01 = RandomForestClassifier(n_estimators=500, n_jobs=-1) #-1代表使用机器最大线程数来运算
rnd_clf01.fit(iris['data'],iris['target'])
for name,score in zip(iris['feature_names'], rnd_clf01.feature_importances_):
    print(name, score)