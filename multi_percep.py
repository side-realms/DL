import numpy as np

class MLP(object):
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.l1 = Layer(input_dim=input_dim, output_dim=hidden_dim, activation=sigmoid, dactivation=dsigmoid)
        self.l2 = Layer(input_dim=hidden_dim, output_dim=output_dim, activation=sigmoid, dactivation=dsigmoid)
        self.Layers = [self.l1, self.l2]

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        h = self.l1(x)
        y = self.l2(h)
        return y


class Layer(object):
    # 重みとバイアス w, b の初期化
    # 活性化関数の定義（シグモイド？ソフトマックス？）
    # 活性化関数の微分（誤差の逆伝搬）
    def __init__(self, input_dim, output_dim, activation, dactivation): 
        #self.w = np.zeros((input_dim, output_dim))
        self.w = np.random.normal(input_dim, output_dim)
        self.b = np.zeros(output_dim)
        self.activation = activation
        self.dactivation = dactivation

    # forward を呼び出す用
    # Layer() って呼び出せば実行される
    def __call__(self, x):
        return self.forward(x)

    # 順伝搬の関数(入力 x に対して活性化関数を通す)
    def forward(self,x):
        self.input = x
        return self.activation(np.dot(x, self.w) + self.b)

    # 誤差を返す
    def backward(self, delta, w):
        return self.dactivation((np.dot(self.input, w.T) + self.b)*np.dot(delta, w.T))

    # 誤差 dw, db を出力
    def gradients(self, delta):
        dw = np.dot(self.input.T, delta)
        db = np.dot(np.ones(self.input.shape[0]), delta)
        return dw, db
    
def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))

if __name__ == '__main__':
    np.random.seed(123)
    rate = 0.01

    x = np.array([[0,0], [0,1], [1,0], [1,1]]) #XOR
    t = np.array([[0],[1],[1],[0]])

    model = MLP(2,2,1)

    def loss(t, y):
        return(-t*np.log(y)-(1-t)*np.log(1-y)).sum()
    
    def train(x,t):
        y=model(x)
        for i, layer in enumerate(model.Layers[::-1]):#(self.l1, self.l2)
            if i == 0:
                # 出力層
                delta = y-t
            else:
                delta = layer.backward(delta, w)
            dw, db = layer.gradients(delta)
            layer.w = layer.w - rate*dw
            layer.b = layer.b - rate*db
            w = layer.w
        Loss = loss(t,y)
        return Loss
    
    epochs = 1000

    for epoch in range(epochs):
        train_loss = train(x, t)

        if epoch % 100 == 0 or epoch == epochs - 1:
            print('epoch: {}, loss: {}'.format(
                epoch+1,
                train_loss
            ))

    for input in x:
        print('{} => {:.3f}'.format(input, model(input)[0]))