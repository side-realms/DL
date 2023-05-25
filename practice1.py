import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

np.random.seed(123)
tf.random.set_seed(123)

# データの準備
# make_moons
# 学習用とトレーニング用
N = 300
noise = 0.5
x,t = datasets.make_moons(N, noise=noise)
t = t.reshape(N,1)
# test_size で何割をテスト用にするか選択する
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.2)

# モデル構築
# keras は model.add でネットワーク層を追加していくやり方ができるらしい
# model.Sequential

model = Sequential() # 系列モデル，層を積み重ねることができる
model.add(Dense(3,activation='sigmoid')) # Dense：個々のパーセプトロンが次のレイヤーのすべてに繋がる
model.add(Dense(1,activation='sigmoid')) # 最小構成がそれぞれ 3, 1

# モデル学習
# モデルの学習率，誤差関数，評価指標など
# model.fit() で学習

# metrics=['accuracy'] これをつけないと acc が評価軸に導入されない
model.compile(optimizer='sgd', loss='mean_squared_error',metrics=['accuracy'])
model.fit(x_train, t_train, epochs=100, batch_size=30)

# モデル評価
# model.evaluate

loss, acc = model.evaluate(x_test, t_test)
print(loss)
print(acc)
