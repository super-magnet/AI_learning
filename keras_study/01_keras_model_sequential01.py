from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation

layers = [Dense(32, input_shape=(784,)),
          Activation('relu'),
          Dense(10),
          Activation('softmax')
          ]

model = Sequential(layers)
model.summary()