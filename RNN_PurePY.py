import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, seq_len, lr=1e-2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.lr = lr

        # 初始化权重（小随机值）
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.1  # input -> hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.1  # hidden -> hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.1  # hidden -> output
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs, h0=None):
        """
        inputs: shape (seq_len, input_size)  (numpy)
        returns: outputs list (seq_len) and hidden states list (seq_len+1)
        """
        if h0 is None:
            h_prev = np.zeros((self.hidden_size, 1))
        else:
            h_prev = h0
        hs = [h_prev]
        ys = []

        for t in range(self.seq_len):
            x = inputs[t].reshape(-1,1)  # column
            h = np.tanh(self.Wxh @ x + self.Whh @ h_prev + self.bh)  # hidden
            y = self.Why @ h + self.by  # linear output
            hs.append(h)
            ys.append(y)
            h_prev = h
        return ys, hs

    def compute_loss_and_grads(self, inputs, targets):
        """
        targets: list of shape (seq_len, output_size) or array
        returns loss and grads dict
        """
        ys, hs = self.forward(inputs)
        # prepare grads
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        loss = 0.0
        # MSE over sequence, and gradients w.r.t outputs
        dht_next = np.zeros((self.hidden_size, 1))
        for t in reversed(range(self.seq_len)):
            y = ys[t]  # (output,1)
            target = targets[t].reshape(-1,1)
            diff = y - target
            loss += 0.5 * np.sum(diff**2)

            # gradient w.r.t output weights
            dWhy += diff @ hs[t+1].T
            dby += diff

            # backprop into hidden state
            dh = self.Why.T @ diff + dht_next  # (hidden,1)
            # backprop through tanh
            h_raw = hs[t+1]
            dh_raw = (1 - h_raw * h_raw) * dh  # derivative tanh
            dbh += dh_raw
            dWxh += dh_raw @ inputs[t].reshape(1,-1)
            dWhh += dh_raw @ hs[t].T
            dht_next = self.Whh.T @ dh_raw

        # clip gradients to avoid exploding
        for g in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(g, -5, 5, out=g)

        grads = {
            'Wxh': dWxh, 'Whh': dWhh, 'Why': dWhy,
            'bh': dbh, 'by': dby
        }
        return loss, grads

    def step(self, grads):
        # SGD update
        self.Wxh -= self.lr * grads['Wxh']
        self.Whh -= self.lr * grads['Whh']
        self.Why -= self.lr * grads['Why']
        self.bh  -= self.lr * grads['bh']
        self.by  -= self.lr * grads['by']


# Toy dataset: predict next value of sine sequence
def generate_sine_sequences(num_seqs=1000, seq_len=10, noise=0.0):
    X = []
    Y = []
    for _ in range(num_seqs):
        start = np.random.rand() * 2 * np.pi
        xs = [np.sin(start + i * 0.1) + noise * np.random.randn() for i in range(seq_len+1)]
        # inputs are first seq_len, target is next value for each step (shifted)
        X.append(np.array(xs[:-1]).reshape(seq_len,1))
        Y.append(np.array(xs[1:]).reshape(seq_len,1))
    return np.array(X), np.array(Y)

if __name__ == '__main__':
    np.random.seed(42)
    seq_len = 20
    rnn = SimpleRNN(input_size=1, hidden_size=32, output_size=1, seq_len=seq_len, lr=1e-2)

    X, Y = generate_sine_sequences(num_seqs=200, seq_len=seq_len, noise=0.0)
    epochs = 200
    batch_size = 1  # our simple impl uses batch=1

    for ep in range(1, epochs+1):
        total_loss = 0.0
        perm = np.random.permutation(len(X))
        for idx in perm:
            x = X[idx]  # shape (seq_len,1)
            y = Y[idx]  # shape (seq_len,1)
            loss, grads = rnn.compute_loss_and_grads(x, y)
            rnn.step(grads)
            total_loss += loss
        if ep % 10 == 0 or ep == 1:
            print(f"Epoch {ep:03d}, avg loss: {total_loss/len(X):.6f}")

    # Test: feed one sequence and predict next-step sequence
    test_x = X[0]
    ys_pred, _ = rnn.forward(test_x)
    pred_seq = np.array([y.ravel()[0] for y in ys_pred])
    true_seq = Y[0].ravel()
    print("true next values (first 10):", np.round(true_seq[:10], 3))
    print("pred next values (first 10):", np.round(pred_seq[:10], 3))
