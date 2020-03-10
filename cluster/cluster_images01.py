#KMeans聚类用来压缩图片
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def restore_image(cl_center, cluster, shape):
    # cl_center代表聚出的类的中心，是一个长度为簇个数的一维数组，元素是各个簇中心坐标；
    # cluster代表聚类结果，是一个长度为像素数的一维数组，元素是各个像素所属簇的序号
    row, column, rgb = shape
    image = np.empty((row,column,3))  #内括号内部表示维度
    index = 0
    for i in range(row):
        for j in range(column):
            image[i,j] = cl_center[cluster[index]]
            index += 1
    return image

def show_scatter(a):
    N = 10
    print('原始数据：\n', a)
    density, edges = np.histogramdd(a, bins=[N,N,N], range=[(0,1),(0,1),(0,1)])
    density /= density.max()
    x=y=z = np.arange(N)
    d = np.meshgrid(x,y,z)

    fig = plt.figure(1, facecolor='w')
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(d[1], d[0], d[2], c='r', s=100*density, marker='o', depthshade=True)
    ax.set_xlabel(u'红色分量')
    ax.set_ylabel(u'绿色分量')
    ax.set_zlabel(u'蓝色分量')
    plt.title(u'图像颜色三维频数分布', fontsize=20)

    plt.figure(2, facecolor='w')
    den = density[density>0]
    den = np.sort(den[::-1])
    t = np.arange(len(den))
    plt.plot(t, den, 'r-', t, den, 'go', lw=2)
    plt.title(u'图像颜色频数分布', fontsize=18)
    plt.grid(True)

    plt.show()

if __name__ =='__main__':
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    num_vq = 256
    im = Image.open('../data/Lena.png')
    image = np.array(im).astype(np.float) / 255 #把像素数据值缩放到0-1
    image = image[:,:,:3]
    image_v = image.reshape((-1,3))
    model = KMeans(num_vq)
    show_scatter(image_v)

    N = image_v.shape[0]  #图像像素总数
    # 随机选择1000个样本进行聚类
    idx = np.random.randint(0, N, size=1000)
    image_sample = image_v[idx]
    model.fit(image_sample)
    c = model.predict(image_v)
    print('聚类结果：\n', c)
    print('聚类中心：\n', model.cluster_centers_)

    plt.figure(figsize=(15,8),facecolor='w')
    plt.subplot(121)
    plt.axis('off')
    plt.title(u'原始图片', fontsize=18)
    plt.imshow(image)

    plt.subplot(122)
    vq_image = restore_image(model.cluster_centers_, c, image.shape)
    plt.axis('off')
    plt.title(u'矢量量化后图片：%d色' % num_vq, fontsize=20)
    plt.imshow(vq_image)

    plt.tight_layout(1,2)
    plt.show()





