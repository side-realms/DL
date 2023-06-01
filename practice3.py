from tensorflow.keras import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

msint = datasets.mnist
(x_train, t_train), (x_test, t_test) = msint.load_data()

x_train = (x_train.reshape(-1, 784)/255).astype(np.float32)
x_test = (x_test.reshape(-1, 784)/255).astype(np.float32)

t_train = np.eye(10)[t_train].astype(np.float32)
t_test = np.eye(10)[t_test].astype(np.float32)

model = Sequential()
model.add(Dense(200, activation='sigmoid'))
model.add(Dense(200, activation='sigmoid'))
model.add(Dense(200, activation='sigmoid'))
model.add(Dense(200, activation='sigmoid'))
model.add(Dense(200, activation='sigmoid'))
model.add(Dense(200, activation='sigmoid'))
model.add(Dense(200, activation='sigmoid'))
model.add(Dense(200, activation='sigmoid'))
model.add(Dense(200, activation='sigmoid'))
model.add(Dense(200, activation='sigmoid'))
#model.add(Dense(200, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, t_train, epochs=10, batch_size=100, verbose=2)
loss, acc = model.evaluate(x_test, t_test, verbose=0)
print(acc)
