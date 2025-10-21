import numpy as np
from tensorflow.keras.datasets import mnist  # 仅用于加载数据，训练仍用 NumPy

# ====== CNN Layer Implementations ======

class Conv3x3:
    """3x3 Convolution Layer"""
    def __init__(self, num_filters):
        self.num_filters = num_filters
        # Initialize filters with small random values
        self.filters = np.random.randn(num_filters, 3, 3) * 0.1
        self.last_input = None
    
    def forward(self, input_img):
        """Forward pass through convolution"""
        self.last_input = input_img
        h, w = input_img.shape
        output = np.zeros((self.num_filters, h - 2, w - 2))
        
        for f in range(self.num_filters):
            for i in range(h - 2):
                for j in range(w - 2):
                    output[f, i, j] = np.sum(input_img[i:i+3, j:j+3] * self.filters[f])
        
        return output

class ReLU:
    """ReLU Activation Layer"""
    def __init__(self):
        self.last_input = None
    
    def forward(self, input_data):
        self.last_input = input_data
        return np.maximum(0, input_data)

class MaxPool2:
    """2x2 Max Pooling Layer"""
    def __init__(self):
        self.last_input = None
    
    def forward(self, input_data):
        self.last_input = input_data
        num_filters, h, w = input_data.shape
        output = np.zeros((num_filters, h // 2, w // 2))
        
        for f in range(num_filters):
            for i in range(0, h, 2):
                for j in range(0, w, 2):
                    output[f, i//2, j//2] = np.max(input_data[f, i:i+2, j:j+2])
        
        return output

class Softmax:
    """Softmax Layer for Classification"""
    def __init__(self, input_len, nodes):
        # Initialize weights and biases
        self.weights = np.random.randn(input_len, nodes) * 0.1
        self.biases = np.zeros(nodes)
        self.last_input = None
        self.last_totals = None
    
    def forward(self, input_data):
        self.last_input = input_data
        input_data = input_data.flatten()
        
        totals = np.dot(input_data, self.weights) + self.biases
        self.last_totals = totals
        
        # Softmax activation
        exp_vals = np.exp(totals - np.max(totals))
        return exp_vals / np.sum(exp_vals)

# 加载 MNIST（只使用少量样本以便快速演示）
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.astype(np.float32) / 255.0
test_images  = test_images.astype(np.float32) / 255.0

# 只取前 N 个样本来快速训练
N_train = 1000
N_test  = 200
X_train = train_images[:N_train]
y_train = train_labels[:N_train]
X_test  = test_images[:N_test]
y_test  = test_labels[:N_test]

# 创建特征提取器（随机初始化卷积核）
conv = Conv3x3(8)
relu = ReLU()
pool = MaxPool2()
softmax = Softmax(8 * 13 * 13, 10)

# 将一张图片通过 conv+relu+pool 得到特征向量
def extract_feature(img):
    out = conv.forward(img)   # shape (num_filters, 26, 26)
    out = relu.forward(out)
    out = pool.forward(out)   # shape (num_filters,13,13)
    return out.flatten()      # shape (8*13*13,)

# 预计算所有训练样本的特征（速度换空间）
feat_train = np.zeros((N_train, conv.num_filters * 13 * 13))
for i in range(N_train):
    feat_train[i] = extract_feature(X_train[i])

feat_test = np.zeros((N_test, conv.num_filters * 13 * 13))
for i in range(N_test):
    feat_test[i] = extract_feature(X_test[i])

# 简单的 Softmax 训练（交叉熵 + 梯度下降）
lr = 0.1
epochs = 10
W = softmax.weights.copy()   # shape (input_len, 10)
b = softmax.biases.copy()     # shape (10,)

def softmax_forward_logits(x, W, b):
    logits = x @ W + b
    ex = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = ex / np.sum(ex, axis=1, keepdims=True)
    return probs, logits

def one_hot(labels, C=10):
    y = np.zeros((labels.size, C))
    y[np.arange(labels.size), labels] = 1
    return y

y_onehot = one_hot(y_train, 10)

for ep in range(epochs):
    # 简单批量训练（全量小批次）
    probs, logits = softmax_forward_logits(feat_train, W, b)
    loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-12), axis=1))
    # grads for logits: (probs - y) / N
    dlogits = (probs - y_onehot) / N_train
    dW = feat_train.T @ dlogits   # shape (input_len, 10)
    db = np.sum(dlogits, axis=0)

    # update
    W -= lr * dW
    b -= lr * db

    # 计算训练准确率
    preds = np.argmax(probs, axis=1)
    acc = np.mean(preds == y_train)
    print(f"Epoch {ep+1}/{epochs} - loss: {loss:.4f} - acc: {acc:.4f}")

# 在测试集上评估
probs_test, _ = softmax_forward_logits(feat_test, W, b)
test_preds = np.argmax(probs_test, axis=1)
test_acc = np.mean(test_preds == y_test)
print("Test acc:", test_acc)
