# coding:utf-8

import numpy as np
from sklearn import linear_model, datasets
import  matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import fft
from scipy.io import wavfile

#准备音乐数据
'''def create_fft(g,n):
    rad = r'F:/CS/python/2019.06/音乐数据/genres/' + g + '/converted/' + g + '.' + str(n).zfill(5)+'.au.wav'
    sample_rate, X = wavfile.read(rad)
    fft_features = abs(fft(X))[:1000]
    sad = 'F:/CS/python/2019.06/音乐数据/trainset/' + g + '.' + str(n).zfill(5) + '.fft'
    np.save(sad, fft_features)

# create fft
genre_list = ['classical','jazz','country','pop','rock','metal']
for g in genre_list:
    for n in range(100):
        create_fft(g, n)
'''
# 加载训练数据集，分割训练集以及测试集，进行分类器的训练
# 构造训练集
# read fft
genre_list = ['classical','jazz','country','pop','rock','metal']
X = []
y = []
for g in genre_list:
    for n in range(100):
        rad = 'F:/CS/python/2019.06/音乐数据/trainset/' + g + '.' + str(n).zfill(5) + '.fft.npy'
        ftt_features = np.load(rad)
        X.append(ftt_features)
        y.append(genre_list.index(g))

# 采用np.array()函数将python中序列list格式数据，转换为机器学习适配的矩阵array形式
X = np.array(X)
y = np.array(y)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(multi_class='ovr' )
model.fit(X,y)

# 可以采用Python内建的持久性模型pickle来保存scikit的模型
'''
import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
clf2.predict(X[0])
'''

# prepare test  data
sample_rate, test = wavfile.read('F:/CS/python/2019.06/音乐数据/trainset/sample/mono.wav')
testdata_fft_features = abs(fft(test))[:1000]
print(sample_rate, testdata_fft_features, len(testdata_fft_features))
type_index = model.predict([testdata_fft_features])[0]
print(type_index)
print(genre_list[type_index])



