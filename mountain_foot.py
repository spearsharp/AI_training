import numpy as np

W = np.random.randn(10, 3073) * 0.001    # generate a random initial number
best_loss = float('inf')
L = lambda X, Y, W: np.sum((X.dot(W.T) - Y) ** 2)  # Example loss function
Xtr_cols = np.random.randn(500, 3073)  # Example training data
Ytr = np.random.randn(500, 10)         # Example training labels
for i in range(1000000):
	step_size = 0.0001
	Wtry = W + np.random.randn(10, 3073) * step_size
	loss = L(Xtr_cols, Ytr, Wtry)
	if loss < best_loss:
		W = Wtry
		best_loss = loss
	print('iter %d loss is %f' % (i, best_loss))
