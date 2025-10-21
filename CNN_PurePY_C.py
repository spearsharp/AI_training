import numpy as np

# -----------------------------
# 1️⃣ 卷积层实现
# -----------------------------
class Conv3x3:
    def __init__(self, num_filters):
        self.num_filters = num_filters
        # 初始化卷积核: (num_filters, 3, 3)
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        h, w = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i+3), j:(j+3)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input
        h, w = input.shape
        output = np.zeros((self.num_filters, h - 2, w - 2))
        for im_region, i, j in self.iterate_regions(input):
            output[:, i, j] = np.sum(im_region * self.filters, axis=(1,2))
        return output

# -----------------------------
# 2️⃣ ReLU 激活层
# -----------------------------
class ReLU:
    def forward(self, input):
        self.last_input = input
        return np.maximum(0, input)

# -----------------------------
# 3️⃣ 最大池化层
# -----------------------------
class MaxPool2:
    def iterate_regions(self, image):
        h, w = image.shape
        new_h = h // 2
        new_w = w // 2
        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i*2):(i*2+2), (j*2):(j*2+2)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input
        num_filters, h, w = input.shape
        output = np.zeros((num_filters, h // 2, w // 2))
        for f in range(num_filters):
            for im_region, i, j in self.iterate_regions(input[f]):
                output[f, i, j] = np.max(im_region)
        return output

# -----------------------------
# 4️⃣ 全连接层 + Softmax
# -----------------------------
class Softmax:
    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input
        totals = np.dot(input, self.weights) + self.biases
        exp = np.exp(totals - np.max(totals))
        return exp / np.sum(exp, axis=0)

# -----------------------------
# 🧩 构建网络并测试前向传播
# -----------------------------
if __name__ == "__main__":
    np.random.seed(42)
    
    # 模拟单通道 28x28 图像
    image = np.random.randn(28, 28)

    # 创建网络
    conv = Conv3x3(8)       # 8 个卷积核
    relu = ReLU()
    pool = MaxPool2()
    softmax = Softmax(8 * 13 * 13, 10)  # 输出10类（如0-9）

    # 前向传播
    out = conv.forward(image)
    out = relu.forward(out)
    out = pool.forward(out)
    out = softmax.forward(out)

    print("✅ CNN forward output shape:", out.shape)
    print("Predicted class:", np.argmax(out))
