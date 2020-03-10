import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

# 先读入数据
(X_train, y_train), (X_test, y_test) = mnist.load_data('mnist.npz ')
# 看一下数据集的样子
print(X_train[0].shape)
print(y_train[0])

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

X_train /= 255
X_test /= 255

def y_t(y):
    y_ohot = np.zeros(10)
    y_ohot[y] = 1
    return y_ohot

# y_train_ohot = np.array([y_t(y_train[i]) for i in range(len(y_train))])
y_train_ohot = np.array([y_t(i) for i in y_train])
y_test_ohot = np.array([y_t(i) for i in y_test])

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=[3,3], strides=[1,1], padding='same', input_shape=[28,28,1], activation='relu'))
model.add(MaxPooling2D(pool_size=[2,2]))
model.add(Dropout(0.5))

model.add(Conv2D(filters=128, kernel_size=[3,3], strides=[1,1], padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=[2,2]))
model.add(Dropout(0.5))

model.add(Conv2D(filters=256, kernel_size=[3,3], strides=[1,1], padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=[2,2]))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_ohot, validation_data=(X_test, y_test_ohot), batch_size=128, epochs=1)
scores = model.evaluate(X_test, y_test_ohot, verbose=0)
print(scores)