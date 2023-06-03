from tensorflow.keras import datasets, optimizers
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

msint = datasets.mnist
(x_train, t_train), (x_test, t_test) = msint.load_data()

x_train = (x_train.reshape(-1, 784)/255).astype(np.float32)
x_test = (x_test.reshape(-1, 784)/255).astype(np.float32)

x_train, x_val, t_train, t_val = train_test_split(x_train, t_train, test_size=0.2)

optimizer = optimizers.SGD(learning_rate=0.1, momentum=0.9)

model = Sequential()
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
# val_loss を監視し続け，5 回連続で上回れば早期終了

hist = model.fit(x_train, t_train, epochs=1000, batch_size=100, verbose=2, validation_data=(x_val, t_val), callbacks=[es])

val_loss = hist.history['val_loss']

fig = plt.figure()
plt.rc('font', family='serif')
plt.plot(range(len(val_loss)), val_loss,
         color='black', linewidth=1)
plt.xlabel('epochs')
plt.ylabel('loss')
# plt.savefig('output.jpg')
plt.show()

loss, acc = model.evaluate(x_test, t_test, verbose=0)
print(acc)
