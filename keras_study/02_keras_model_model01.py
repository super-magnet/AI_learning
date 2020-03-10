from keras.models import Model
from keras.layers import Dense
from keras.layers import Activation

input = Input(shape(784,))

x = Dense(32, activation='relu')(input)
x = Dense(64, activation='relu')(input)
y = Dense(10, activation='softmax')(x)

model = Model(inputs=input, outputs=y)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(data, labels)