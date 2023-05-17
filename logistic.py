# ロジェスティック回帰

import numpy as np

class Logistic(object):
    def __init__(self, d):
        # まずは w, b を初期化する
        self.d = d # 次元
        self.w = np.zeros(d) # 重み w
        self.b = 0 # バイアス b

    def calc(self, x): # モデルの出力
        # y=wx+b
        # sigmoid
        y = np.dot(x, self.w.T) + self.b
        return (1/(1+np.exp(-y)))
    
    def learn(self, x, t):
        # delta = y-t
        # dw = xt(y-t)
        # db = 1t(y-t)
        y = self.calc(x)
        delta = y-t
        dw = np.dot(x.T, delta)
        db = np.dot(np.ones(x.shape[0]), delta)
        return (dw, db)
    
    

if __name__ == '__main__':
    np.random.seed(123)

    dim = 2
    N = 20
    n = 0.01
    epochs = 100

    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    t = np.array([0, 1, 1, 1])

    model = Logistic(d=dim) # インスタンス

    def loss(y, t):
        # 誤差 -(t logy + (1-t)log(1-y))
        return(t*np.log(y) - (1-t)*np.log(1-y)).sum()
    
    def train(x, t):
        dw, db = model.learn(x, t)
        model.w = model.w - n*(dw)
        model.b = model.b - n*(db)
        return(loss(model.calc(x), t))
    

    for epoch in range(epochs):
        train_loss = train(x, t)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print('epoch: {}, loss: {:.3f}'.format(
                epoch+1,
                train_loss
            ))

    for input in x:
        print('{} => {:.3f}'.format(input, model.calc(input)))
