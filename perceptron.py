# 単純パーセプトロン

import numpy as np

class Perceptron(object):
    def __init__(self, d):
        # まずは w, b を初期化する
        self.d = d # 次元
        self.w = np.zeros(d) # 重み w
        self.b = 0 # バイアス b

    def calc(self, x):
        # y=f(wx+b)
        y = np.dot(self.w.T, x)
        return(y>0) # y>0 なら 1 を返す
    
    def learn(self, x, t):
        y = self.calc(x)
        delta = y-t
        w = delta * x # w=(y-t)x
        b = delta # b=(y-t)
        return (w, b)
        

if __name__ == '__main__':
    np.random.seed(123)

    dim = 2
    N = 20

    x1 = np.random.normal(0, 0.5, (N//2, dim))
    x2 = np.random.normal(20, 0.5, (N//2, dim))
    x = np.concatenate((x1, x2), axis=0) # 入力

    t1 = np.zeros(N//2)
    t2 = np.ones(N//2)
    t = np.concatenate((t1, t2)) # 出力

    model = Perceptron(d=dim) # インスタンス

    while True:
        for i in range(N):
            dw, db = model.learn(x[i], t[i])
            model.w = model.w - dw
            model.b = model.b - db
        if ((dw==0).all() and (db==0).all()):
            break
    
    print('w:', model.w)  # => w: [1.660725   1.49465147]
    print('b:', model.b)  # => b: -10.0

    print('(0, 0) =>', model.calc([0, 0]))  # => 0
    print('(10, 10) =>', model.calc([1, 1]))  # => 1

    # モデル学習
        # 誤差が0かの判定
        # パラメータの更新(w=w-Δ，b同じ)
    # モデル評価
        # w, b が得られるはずなので，