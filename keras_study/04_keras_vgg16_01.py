from keras.applications.vgg16 import  VGG16
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import cv2

#复用模型VGG，并更改掉最后一层参数和连接
model_vgg_16 = VGG16(input_shape = (48,48,3), weights='imagenet', include_top = False)
for layer in model_vgg_16.layers:
    layer.trainable = False
model = Flatten()(model_vgg_16.output)
model = Dense(512, activation='relu', name='fc1')(model)
model = Dense(512, activation='relu', name='fc2')(model)
model = Dropout(0.5)(model)
model = Dense(10, activation='softmax', name='fc3')(model)
model_vgg = Model(inputs = model_vgg_16.input, outputs = model, name='alter_vgg16')

model_vgg.summary()
sgd = SGD(lr=0.05, decay=1e-5)
model_vgg.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
#使用open_cv设置输入数据的格式
(X_train, y_train),(X_test, y_test) = mnist.load_data('mnist.npz')
X_train, y_train = X_train[:10000], y_train[:10000]
X_test, y_test = X_test[:10000], y_test[:10000]
X_train = [cv2.cvtColor(cv2.resize(X, 48*48), cv2.COLOR_GRAY2RGB) for X in X_train]
X_test = [cv2.cvtColor(cv2.resize(X, 48*48), cv2.COLOR_GRAY2RGB) for X in X_test]

print('X_train.shape', X_train.shape)
print('X_test.shape', X_test.shape)
X_train /= 255
X_test /= 255

def y_onehot(y):
    y_oo = np.zeros(10)
    y_oo[y] = 1
    return y_oo

y_train_oo = [y_onehot(y) for y in y_train]
y_test_oo = [y_onehot(y) for y in y_test]

model_vgg.fit(x=X_train, y=y_train_oo, validation_data=(X_test, y_test_oo), batch_size=128, epochs=1)
