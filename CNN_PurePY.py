"""
纯 NumPy 实现的卷积神经网络（CNN）
从零开始实现所有组件，不使用任何深度学习框架
包含：卷积层、池化层、全连接层、激活函数、反向传播
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

print("=" * 70)
print("纯 NumPy 实现的卷积神经网络 (CNN)")
print("=" * 70)

# ==================== 1. 激活函数及其导数 ====================
class Activation:
    """激活函数类"""
    
    @staticmethod
    def relu(x):
        """ReLU 激活函数"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        """ReLU 导数"""
        return (x > 0).astype(float)
    
    @staticmethod
    def softmax(x):
        """Softmax 激活函数"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def sigmoid(x):
        """Sigmoid 激活函数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x):
        """Sigmoid 导数"""
        s = Activation.sigmoid(x)
        return s * (1 - s)


# ==================== 2. 卷积层 ====================
class ConvLayer:
    """卷积层"""
    
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1):
        """
        初始化卷积层
        
        参数:
            input_channels: 输入通道数
            output_channels: 输出通道数（滤波器数量）
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
        """
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # He 初始化权重
        self.weights = np.random.randn(
            output_channels, input_channels, kernel_size, kernel_size
        ) * np.sqrt(2.0 / (input_channels * kernel_size * kernel_size))
        
        self.bias = np.zeros((output_channels, 1))
        
        # 用于保存中间结果
        self.cache = {}
    
    def add_padding(self, x, pad):
        """添加零填充"""
        if pad == 0:
            return x
        return np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入 shape (batch_size, channels, height, width)
        返回:
            输出 shape (batch_size, output_channels, out_height, out_width)
        """
        batch_size, channels, height, width = x.shape
        
        # 保存输入用于反向传播
        self.cache['input'] = x
        
        # 添加填充
        x_padded = self.add_padding(x, self.padding)
        
        # 计算输出尺寸
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # 初始化输出
        output = np.zeros((batch_size, self.output_channels, out_height, out_width))
        
        # 卷积操作
        for b in range(batch_size):
            for c_out in range(self.output_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # 提取区域
                        region = x_padded[b, :, h_start:h_end, w_start:w_end]
                        
                        # 卷积计算
                        output[b, c_out, h, w] = np.sum(
                            region * self.weights[c_out]
                        ) + self.bias[c_out]
        
        return output
    
    def backward(self, dout, learning_rate=0.01):
        """
        反向传播
        
        参数:
            dout: 上游梯度
            learning_rate: 学习率
        返回:
            dx: 输入的梯度
        """
        x = self.cache['input']
        batch_size, channels, height, width = x.shape
        x_padded = self.add_padding(x, self.padding)
        
        _, _, out_height, out_width = dout.shape
        
        # 初始化梯度
        dx_padded = np.zeros_like(x_padded)
        dw = np.zeros_like(self.weights)
        db = np.zeros_like(self.bias)
        
        # 计算梯度
        for b in range(batch_size):
            for c_out in range(self.output_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        region = x_padded[b, :, h_start:h_end, w_start:w_end]
                        
                        # 权重梯度
                        dw[c_out] += region * dout[b, c_out, h, w]
                        
                        # 输入梯度
                        dx_padded[b, :, h_start:h_end, w_start:w_end] += \
                            self.weights[c_out] * dout[b, c_out, h, w]
                
                # 偏置梯度
                db[c_out] = np.sum(dout[:, c_out, :, :])
        
        # 更新参数
        self.weights -= learning_rate * dw / batch_size
        self.bias -= learning_rate * db.reshape(-1, 1) / batch_size
        
        # 去除填充
        if self.padding > 0:
            dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dx = dx_padded
        
        return dx


# ==================== 3. 池化层 ====================
class MaxPoolLayer:
    """最大池化层"""
    
    def __init__(self, pool_size=2, stride=2):
        """
        初始化池化层
        
        参数:
            pool_size: 池化窗口大小
            stride: 步长
        """
        self.pool_size = pool_size
        self.stride = stride
        self.cache = {}
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入 shape (batch_size, channels, height, width)
        返回:
            输出
        """
        batch_size, channels, height, width = x.shape
        
        # 保存输入
        self.cache['input'] = x
        
        # 计算输出尺寸
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        
        # 初始化输出
        output = np.zeros((batch_size, channels, out_height, out_width))
        
        # 保存最大值位置
        self.cache['max_indices'] = {}
        
        # 池化操作
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size
                        
                        region = x[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, h, w] = np.max(region)
                        
                        # 保存最大值位置
                        max_idx = np.unravel_index(np.argmax(region), region.shape)
                        self.cache['max_indices'][(b, c, h, w)] = (
                            h_start + max_idx[0], w_start + max_idx[1]
                        )
        
        return output
    
    def backward(self, dout):
        """
        反向传播
        
        参数:
            dout: 上游梯度
        返回:
            dx: 输入的梯度
        """
        x = self.cache['input']
        dx = np.zeros_like(x)
        
        batch_size, channels, out_height, out_width = dout.shape
        
        # 将梯度传递到最大值位置
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        max_h, max_w = self.cache['max_indices'][(b, c, h, w)]
                        dx[b, c, max_h, max_w] += dout[b, c, h, w]
        
        return dx


# ==================== 4. 全连接层 ====================
class FullyConnectedLayer:
    """全连接层"""
    
    def __init__(self, input_size, output_size):
        """
        初始化全连接层
        
        参数:
            input_size: 输入大小
            output_size: 输出大小
        """
        self.input_size = input_size
        self.output_size = output_size
        
        # Xavier 初始化
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))
        
        self.cache = {}
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入 shape (batch_size, input_size)
        返回:
            输出 shape (batch_size, output_size)
        """
        self.cache['input'] = x
        output = np.dot(x, self.weights) + self.bias
        return output
    
    def backward(self, dout, learning_rate=0.01):
        """
        反向传播
        
        参数:
            dout: 上游梯度
            learning_rate: 学习率
        返回:
            dx: 输入的梯度
        """
        x = self.cache['input']
        batch_size = x.shape[0]
        
        # 计算梯度
        dx = np.dot(dout, self.weights.T)
        dw = np.dot(x.T, dout)
        db = np.sum(dout, axis=0, keepdims=True)
        
        # 更新参数
        self.weights -= learning_rate * dw / batch_size
        self.bias -= learning_rate * db / batch_size
        
        return dx


# ==================== 5. Flatten 层 ====================
class FlattenLayer:
    """展平层"""
    
    def __init__(self):
        self.cache = {}
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入 shape (batch_size, channels, height, width)
        返回:
            输出 shape (batch_size, channels * height * width)
        """
        self.cache['input_shape'] = x.shape
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)
    
    def backward(self, dout):
        """
        反向传播
        
        参数:
            dout: 上游梯度
        返回:
            dx: 输入的梯度
        """
        return dout.reshape(self.cache['input_shape'])


# ==================== 6. 损失函数 ====================
class CrossEntropyLoss:
    """交叉熵损失"""
    
    @staticmethod
    def forward(y_pred, y_true):
        """
        计算损失
        
        参数:
            y_pred: 预测值 (batch_size, num_classes)
            y_true: 真实标签 (batch_size,)
        返回:
            损失值
        """
        batch_size = y_pred.shape[0]
        # 防止 log(0)
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        
        # 交叉熵损失
        loss = -np.sum(np.log(y_pred[np.arange(batch_size), y_true])) / batch_size
        return loss
    
    @staticmethod
    def backward(y_pred, y_true):
        """
        计算梯度
        
        参数:
            y_pred: 预测值 (batch_size, num_classes)
            y_true: 真实标签 (batch_size,)
        返回:
            梯度
        """
        batch_size = y_pred.shape[0]
        grad = y_pred.copy()
        grad[np.arange(batch_size), y_true] -= 1
        return grad / batch_size


# ==================== 7. CNN 模型 ====================
class SimpleCNN:
    """简单的 CNN 模型"""
    
    def __init__(self, input_shape=(1, 28, 28), num_classes=10):
        """
        初始化 CNN
        
        参数:
            input_shape: 输入形状 (channels, height, width)
            num_classes: 类别数量
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # 构建网络
        self.layers = OrderedDict()
        
        # 卷积层1: 1 -> 8 通道
        self.layers['conv1'] = ConvLayer(
            input_channels=input_shape[0],
            output_channels=8,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # 池化层1
        self.layers['pool1'] = MaxPoolLayer(pool_size=2, stride=2)
        
        # 卷积层2: 8 -> 16 通道
        self.layers['conv2'] = ConvLayer(
            input_channels=8,
            output_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # 池化层2
        self.layers['pool2'] = MaxPoolLayer(pool_size=2, stride=2)
        
        # Flatten 层
        self.layers['flatten'] = FlattenLayer()
        
        # 计算展平后的大小
        # 28x28 -> 14x14 (pool1) -> 7x7 (pool2)
        flatten_size = 16 * 7 * 7
        
        # 全连接层
        self.layers['fc1'] = FullyConnectedLayer(flatten_size, 128)
        self.layers['fc2'] = FullyConnectedLayer(128, num_classes)
        
        # 激活函数
        self.activation = Activation()
        self.loss_fn = CrossEntropyLoss()
        
        # 训练历史
        self.history = {'loss': [], 'accuracy': []}
    
    def forward(self, x):
        """前向传播"""
        # 卷积层1 + ReLU
        out = self.layers['conv1'].forward(x)
        out = self.activation.relu(out)
        self.layers['conv1'].cache['output'] = out
        
        # 池化层1
        out = self.layers['pool1'].forward(out)
        
        # 卷积层2 + ReLU
        out = self.layers['conv2'].forward(out)
        out = self.activation.relu(out)
        self.layers['conv2'].cache['output'] = out
        
        # 池化层2
        out = self.layers['pool2'].forward(out)
        
        # Flatten
        out = self.layers['flatten'].forward(out)
        
        # 全连接层1 + ReLU
        out = self.layers['fc1'].forward(out)
        out = self.activation.relu(out)
        self.layers['fc1'].cache['output'] = out
        
        # 全连接层2
        out = self.layers['fc2'].forward(out)
        
        # Softmax
        out = self.activation.softmax(out)
        
        return out
    
    def backward(self, dout, learning_rate=0.01):
        """反向传播"""
        # 全连接层2
        dout = self.layers['fc2'].backward(dout, learning_rate)
        
        # ReLU 导数
        fc1_output = self.layers['fc1'].cache['output']
        dout = dout * self.activation.relu_derivative(fc1_output)
        
        # 全连接层1
        dout = self.layers['fc1'].backward(dout, learning_rate)
        
        # Flatten
        dout = self.layers['flatten'].backward(dout)
        
        # 池化层2
        dout = self.layers['pool2'].backward(dout)
        
        # ReLU 导数
        conv2_output = self.layers['conv2'].cache['output']
        dout = dout * self.activation.relu_derivative(conv2_output)
        
        # 卷积层2
        dout = self.layers['conv2'].backward(dout, learning_rate)
        
        # 池化层1
        dout = self.layers['pool1'].backward(dout)
        
        # ReLU 导数
        conv1_output = self.layers['conv1'].cache['output']
        dout = dout * self.activation.relu_derivative(conv1_output)
        
        # 卷积层1
        dout = self.layers['conv1'].backward(dout, learning_rate)
    
    def train_step(self, x, y, learning_rate=0.01):
        """训练一步"""
        # 前向传播
        y_pred = self.forward(x)
        
        # 计算损失
        loss = self.loss_fn.forward(y_pred, y)
        
        # 计算准确率
        predictions = np.argmax(y_pred, axis=1)
        accuracy = np.mean(predictions == y)
        
        # 反向传播
        grad = self.loss_fn.backward(y_pred, y)
        self.backward(grad, learning_rate)
        
        return loss, accuracy
    
    def predict(self, x):
        """预测"""
        y_pred = self.forward(x)
        return np.argmax(y_pred, axis=1)
    
    def evaluate(self, x, y):
        """评估"""
        y_pred = self.forward(x)
        loss = self.loss_fn.forward(y_pred, y)
        predictions = np.argmax(y_pred, axis=1)
        accuracy = np.mean(predictions == y)
        return loss, accuracy


# ==================== 8. 加载 MNIST 数据 ====================
print("\n[步骤 1] 加载 MNIST 数据...")

from tensorflow import keras
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# 归一化
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# 添加通道维度
X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]

# 使用较小的数据集进行演示
X_train_small = X_train[:1000]
y_train_small = y_train[:1000]
X_test_small = X_test[:200]
y_test_small = y_test[:200]

print(f"训练集: {X_train_small.shape}, 标签: {y_train_small.shape}")
print(f"测试集: {X_test_small.shape}, 标签: {y_test_small.shape}")


# ==================== 9. 训练模型 ====================
print("\n[步骤 2] 创建并训练 CNN 模型...")
print("注意: 纯 NumPy 实现速度较慢，这是正常的\n")

# 创建模型
model = SimpleCNN(input_shape=(1, 28, 28), num_classes=10)

# 训练参数
epochs = 5
batch_size = 32
learning_rate = 0.01

# 训练循环
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    epoch_loss = 0
    epoch_acc = 0
    num_batches = len(X_train_small) // batch_size
    
    for batch in range(num_batches):
        # 获取批次数据
        start_idx = batch * batch_size
        end_idx = start_idx + batch_size
        
        x_batch = X_train_small[start_idx:end_idx]
        y_batch = y_train_small[start_idx:end_idx]
        
        # 训练一步
        loss, acc = model.train_step(x_batch, y_batch, learning_rate)
        
        epoch_loss += loss
        epoch_acc += acc
        
        # 打印进度
        if (batch + 1) % 5 == 0:
            print(f"  Batch {batch + 1}/{num_batches} - "
                  f"Loss: {loss:.4f}, Acc: {acc:.4f}")
    
    # 计算平均值
    avg_loss = epoch_loss / num_batches
    avg_acc = epoch_acc / num_batches
    
    # 在测试集上评估
    test_loss, test_acc = model.evaluate(X_test_small, y_test_small)
    
    print(f"\n训练 - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
    print(f"测试 - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
    
    # 保存历史
    model.history['loss'].append(avg_loss)
    model.history['accuracy'].append(avg_acc)

print("\n✓ 训练完成！")


# ==================== 10. 可视化结果 ====================
print("\n[步骤 3] 可视化结果...")

# 训练历史
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(model.history['loss'], 'b-', linewidth=2, marker='o')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(model.history['accuracy'], 'g-', linewidth=2, marker='o')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Training Accuracy', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('numpy_cnn_training.png', dpi=150, bbox_inches='tight')
print("✓ 训练历史已保存: numpy_cnn_training.png")

# 预测示例
num_samples = 10
predictions = model.predict(X_test_small[:num_samples])

plt.figure(figsize=(15, 3))
for i in range(num_samples):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test_small[i, 0], cmap='gray')
    pred = predictions[i]
    true = y_test_small[i]
    color = 'green' if pred == true else 'red'
    plt.title(f"预测: {pred}\n真实: {true}", color=color, fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.savefig('numpy_cnn_predictions.png', dpi=150, bbox_inches='tight')
print("✓ 预测结果已保存: numpy_cnn_predictions.png")


# ==================== 11. 总结 ====================
print("\n" + "=" * 70)
print("项目总结")
print("=" * 70)
print(f"训练样本: {len(X_train_small):,}")
print(f"测试样本: {len(X_test_small):,}")
print(f"训练轮次: {epochs}")
print(f"最终测试准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"\n网络结构:")
print(f"  - 卷积层1: 1 -> 8 通道, 3x3 卷积核")
print(f"  - 最大池化1: 2x2")
print(f"  - 卷积层2: 8 -> 16 通道, 3x3 卷积核")
print(f"  - 最大池化2: 2x2")
print(f"  - 全连接层1: 784 -> 128")
print(f"  - 全连接层2: 128 -> 10")
print(f"\n实现的组件:")
print(f"  ✓ 卷积层 (ConvLayer)")
print(f"  ✓ 最大池化层 (MaxPoolLayer)")
print(f"  ✓ 全连接层 (FullyConnectedLayer)")
print(f"  ✓ Flatten 层")
print(f"  ✓ 激活函数 (ReLU, Softmax, Sigmoid)")
print(f"  ✓ 损失函数 (CrossEntropyLoss)")
print(f"  ✓ 前向传播")
print(f"  ✓ 反向传播")
print(f"  ✓ 梯度下降优化")
print("=" * 70)
print("\n✓ 完成！纯 NumPy CNN 实现成功运行。")
print("\n注意: 这是教学示例，实际应用建议使用 TensorFlow/PyTorch")
print("=" * 70)