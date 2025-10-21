import numpy as np

# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def dtanh(y):
    return 1 - y ** 2

class SimpleLSTM:
    def __init__(self, input_size, hidden_size, output_size, seq_len, lr=1e-2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.lr = lr

        # 初始化权重
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.Wy = np.random.randn(output_size, hidden_size) * 0.1

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs, h_prev=None, c_prev=None):
        if h_prev is None:
            h_prev = np.zeros((self.hidden_size, 1))
        if c_prev is None:
            c_prev = np.zeros((self.hidden_size, 1))

        hs, cs, ys = [h_prev], [c_prev], []

        for t in range(self.seq_len):
            x = inputs[t].reshape(-1, 1)
            z = np.vstack((h_prev, x))  # 拼接 h_{t-1} 和 x_t

            f = sigmoid(self.Wf @ z + self.bf)
            i = sigmoid(self.Wi @ z + self.bi)
            o = sigmoid(self.Wo @ z + self.bo)
            c_tilde = np.tanh(self.Wc @ z + self.bc)

            c = f * c_prev + i * c_tilde
            h = o * np.tanh(c)
            y = self.Wy @ h + self.by

            hs.append(h)
            cs.append(c)
            ys.append(y)

            h_prev, c_prev = h, c

        return ys, hs, cs

    def compute_loss(self, inputs, targets):
        ys, hs, cs = self.forward(inputs)
        loss = 0.0
        for t in range(self.seq_len):
            y = ys[t]
            target = targets[t].reshape(-1, 1)
            loss += 0.5 * np.sum((y - target) ** 2)
        return loss

# 生成正弦波序列
def generate_sine_sequences(num_seqs=100, seq_len=20):
    X, Y = [], []
    for _ in range(num_seqs):
        start = np.random.rand() * 2 * np.pi
        xs = [np.sin(start + i * 0.1) for i in range(seq_len + 1)]
        X.append(np.array(xs[:-1]).reshape(seq_len, 1))
        Y.append(np.array(xs[1:]).reshape(seq_len, 1))
    return np.array(X), np.array(Y)

if __name__ == '__main__':
    np.random.seed(42)
    seq_len = 20
    lstm = SimpleLSTM(input_size=1, hidden_size=32, output_size=1, seq_len=seq_len, lr=1e-2)
    X, Y = generate_sine_sequences(num_seqs=50, seq_len=seq_len)

    # 前向测试
    loss = lstm.compute_loss(X[0], Y[0])
    print(f"Initial loss: {loss:.4f}")

    ys_pred, _, _ = lstm.forward(X[0])
    pred_seq = np.array([y.ravel()[0] for y in ys_pred])
    true_seq = Y[0].ravel()

    print("True values (first 10):", np.round(true_seq[:10], 3))
    print("Pred values (first 10):", np.round(pred_seq[:10], 3))
