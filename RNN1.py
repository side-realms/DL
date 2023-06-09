import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

# sin 波を予測する
# sin 波を生成し，ノイズを加算する

def  sin(x, T=100):
    data = np.sin(2.0*np.pi* x / T)
    x = np.arange(0, 2*T+1)
    return data + 0.05*np.random.uniform(low=-1.0, high=1.0, size=len(x))

T = 100
f = sin(T).astype(np.float32)
tau = 25
x = []
t = []

for i in range(len(f)-tau):
    x.append(f[i:i+tau])
    print(f[i:i+tau])
    x.append(f[i+tau])

print(np.size(x))

x = np.array(x).reshape(-1, tau, 1)
t = np.array(t).reshape(-1, 1)

# 訓練データと学習データを分類するときはシャッフルしない
x_train, x_val, t_train, t_val = train_test_split(x, t, test_size=0.2, shuffle=False)


# モデルの構築
#  model.add 隠れ層に SimpleRNN を使う
# recurrent_initializer で[過去の隠れ層からの]重みを設定する(orthobonal)
# kernel_initilizer で[入力：隠れ層]の重みを設定する(glorot_normal)
# actiavation = tanh
# 出力層は確率ではなく，そのままの値を出したいので activation=linear

model = Sequential()
model.add(SimpleRNN(50, activation='tahn', kernel_initializer='glorot_normal', recurrent_initializer='orthogonal'))
model.add(Dense(1, activation='linear'))

# モデルの学習
# 誤差は単純な二乗平均誤差

optimizer = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.99, amsgrad=True)
model.compile(optimizer=optimizer, loss='mean_squared_error')

es = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
hist = model.fit(x_train, t_train, epochs=1000, batch_size=100, verbose=2, validation_data=(x_val, t_val), callbacks=[es])


sin = sin(T, ampl=0.)
gen = [None for i in range(tau)]
z = x[:1]
# 逐次的に予測値を求める
for i in range(len(f) - tau):
    preds = model.predict(z[-1:])
    z = np.append(z, preds)[1:]
    z = z.reshape(-1, tau, 1)
    gen.append(preds[0, 0])
# 予測値を可視化
fig = plt.figure()
plt.rc('font', family='serif')
plt.xlim([0, 2*T])
plt.ylim([-1.5, 1.5])
plt.plot(range(len(f)), sin,
         color='gray',
         linestyle='--', linewidth=0.5)
plt.plot(range(len(f)), gen,
         color='black', linewidth=1,
         marker='o', markersize=1, markerfacecolor='black',
         markeredgecolor='black')
# plt.savefig('output.jpg')
plt.show()

# モデル評価
# 




