"""
纯 NumPy 实现循环神经网络 (RNN/LSTM/GRU)
从零开始实现，不使用任何深度学习框架
包含：基础RNN、LSTM、GRU、文本生成示例
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

print("=" * 70)
print("纯 NumPy 实现的循环神经网络 (RNN/LSTM/GRU)")
print("=" * 70)

# ==================== 1. 激活函数 ====================
class Activation:
    """激活函数及其导数"""
    
    @staticmethod
    def tanh(x):
        """双曲正切函数"""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        """tanh 导数"""
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def sigmoid(x):
        """Sigmoid 函数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x):
        """Sigmoid 导数"""
        s = Activation.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def softmax(x):
        """Softmax 函数"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    @staticmethod
    def relu(x):
        """ReLU 函数"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        """ReLU 导数"""
        return (x > 0).astype(float)


# ==================== 2. 基础 RNN 单元 ====================
class VanillaRNNCell:
    """基础 RNN 单元"""
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化 RNN 单元
        
        参数:
            input_size: 输入维度
            hidden_size: 隐藏层维度
            output_size: 输出维度
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 初始化权重 (Xavier 初始化)
        self.Wxh = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.Whh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.Why = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        
        # 偏置
        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, output_size))
        
        # 用于保存中间值
        self.cache = {}
    
    def forward(self, x, h_prev):
        """
        前向传播
        
        参数:
            x: 输入 shape (batch_size, input_size)
            h_prev: 上一时刻的隐藏状态 shape (batch_size, hidden_size)
        返回:
            h_next: 当前时刻的隐藏状态
            y: 输出
        """
        # h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
        h_next = np.tanh(np.dot(x, self.Wxh) + np.dot(h_prev, self.Whh) + self.bh)
        
        # y_t = W_hy * h_t + b_y
        y = np.dot(h_next, self.Why) + self.by
        
        # 保存用于反向传播
        self.cache = {
            'x': x,
            'h_prev': h_prev,
            'h_next': h_next,
            'y': y
        }
        
        return h_next, y
    
    def backward(self, dy, dh_next, learning_rate=0.01):
        """
        反向传播
        
        参数:
            dy: 输出的梯度
            dh_next: 下一时刻隐藏状态的梯度
            learning_rate: 学习率
        返回:
            dh_prev: 上一时刻隐藏状态的梯度
        """
        x = self.cache['x']
        h_prev = self.cache['h_prev']
        h_next = self.cache['h_next']
        
        # 输出层梯度
        dWhy = np.dot(h_next.T, dy)
        dby = np.sum(dy, axis=0, keepdims=True)
        
        # 隐藏层梯度
        dh = np.dot(dy, self.Why.T) + dh_next
        
        # tanh 导数
        dh_raw = dh * (1 - h_next ** 2)
        
        # 权重梯度
        dWxh = np.dot(x.T, dh_raw)
        dWhh = np.dot(h_prev.T, dh_raw)
        dbh = np.sum(dh_raw, axis=0, keepdims=True)
        
        # 上一时刻隐藏状态梯度
        dh_prev = np.dot(dh_raw, self.Whh.T)
        
        # 梯度裁剪（防止梯度爆炸）
        for grad in [dWxh, dWhh, dWhy]:
            np.clip(grad, -5, 5, out=grad)
        
        # 更新权重
        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby
        
        return dh_prev


# ==================== 3. LSTM 单元 ====================
class LSTMCell:
    """LSTM (Long Short-Term Memory) 单元"""
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化 LSTM 单元
        
        参数:
            input_size: 输入维度
            hidden_size: 隐藏层维度
            output_size: 输出维度
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 遗忘门权重
        self.Wf = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.bf = np.zeros((1, hidden_size))
        
        # 输入门权重
        self.Wi = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.bi = np.zeros((1, hidden_size))
        
        # 候选值权重
        self.Wc = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.bc = np.zeros((1, hidden_size))
        
        # 输出门权重
        self.Wo = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.bo = np.zeros((1, hidden_size))
        
        # 输出层权重
        self.Wy = np.random.randn(hidden_size, output_size) * 0.01
        self.by = np.zeros((1, output_size))
        
        self.cache = {}
    
    def forward(self, x, h_prev, c_prev):
        """
        LSTM 前向传播
        
        参数:
            x: 输入
            h_prev: 上一时刻隐藏状态
            c_prev: 上一时刻细胞状态
        返回:
            h_next: 当前隐藏状态
            c_next: 当前细胞状态
            y: 输出
        """
        # 拼接输入和隐藏状态
        concat = np.concatenate([x, h_prev], axis=1)
        
        # 遗忘门: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
        f = Activation.sigmoid(np.dot(concat, self.Wf) + self.bf)
        
        # 输入门: i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
        i = Activation.sigmoid(np.dot(concat, self.Wi) + self.bi)
        
        # 候选值: c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
        c_tilde = np.tanh(np.dot(concat, self.Wc) + self.bc)
        
        # 更新细胞状态: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
        c_next = f * c_prev + i * c_tilde
        
        # 输出门: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
        o = Activation.sigmoid(np.dot(concat, self.Wo) + self.bo)
        
        # 隐藏状态: h_t = o_t ⊙ tanh(c_t)
        h_next = o * np.tanh(c_next)
        
        # 输出
        y = np.dot(h_next, self.Wy) + self.by
        
        # 保存中间值
        self.cache = {
            'x': x, 'h_prev': h_prev, 'c_prev': c_prev,
            'concat': concat, 'f': f, 'i': i, 'c_tilde': c_tilde,
            'c_next': c_next, 'o': o, 'h_next': h_next, 'y': y
        }
        
        return h_next, c_next, y
    
    def backward(self, dy, dh_next, dc_next, learning_rate=0.01):
        """
        LSTM 反向传播
        
        参数:
            dy: 输出梯度
            dh_next: 下一时刻隐藏状态梯度
            dc_next: 下一时刻细胞状态梯度
            learning_rate: 学习率
        返回:
            dh_prev: 上一时刻隐藏状态梯度
            dc_prev: 上一时刻细胞状态梯度
        """
        # 提取缓存值
        concat = self.cache['concat']
        f = self.cache['f']
        i = self.cache['i']
        c_tilde = self.cache['c_tilde']
        c_next = self.cache['c_next']
        c_prev = self.cache['c_prev']
        o = self.cache['o']
        h_next = self.cache['h_next']
        
        # 输出层梯度
        dWy = np.dot(h_next.T, dy)
        dby = np.sum(dy, axis=0, keepdims=True)
        
        # 隐藏状态梯度
        dh = np.dot(dy, self.Wy.T) + dh_next
        
        # 输出门梯度
        do = dh * np.tanh(c_next)
        do_raw = do * o * (1 - o)
        
        # 细胞状态梯度
        dc = dh * o * (1 - np.tanh(c_next) ** 2) + dc_next
        
        # 遗忘门梯度
        df = dc * c_prev
        df_raw = df * f * (1 - f)
        
        # 输入门梯度
        di = dc * c_tilde
        di_raw = di * i * (1 - i)
        
        # 候选值梯度
        dc_tilde = dc * i
        dc_tilde_raw = dc_tilde * (1 - c_tilde ** 2)
        
        # 上一时刻细胞状态梯度
        dc_prev = dc * f
        
        # 权重梯度
        dWf = np.dot(concat.T, df_raw)
        dbf = np.sum(df_raw, axis=0, keepdims=True)
        
        dWi = np.dot(concat.T, di_raw)
        dbi = np.sum(di_raw, axis=0, keepdims=True)
        
        dWc = np.dot(concat.T, dc_tilde_raw)
        dbc = np.sum(dc_tilde_raw, axis=0, keepdims=True)
        
        dWo = np.dot(concat.T, do_raw)
        dbo = np.sum(do_raw, axis=0, keepdims=True)
        
        # 输入梯度
        dconcat = (np.dot(df_raw, self.Wf.T) + 
                   np.dot(di_raw, self.Wi.T) + 
                   np.dot(dc_tilde_raw, self.Wc.T) + 
                   np.dot(do_raw, self.Wo.T))
        
        dh_prev = dconcat[:, self.input_size:]
        
        # 梯度裁剪
        for grad in [dWf, dWi, dWc, dWo, dWy]:
            np.clip(grad, -5, 5, out=grad)
        
        # 更新权重
        self.Wf -= learning_rate * dWf
        self.bf -= learning_rate * dbf
        self.Wi -= learning_rate * dWi
        self.bi -= learning_rate * dbi
        self.Wc -= learning_rate * dWc
        self.bc -= learning_rate * dbc
        self.Wo -= learning_rate * dWo
        self.bo -= learning_rate * dbo
        self.Wy -= learning_rate * dWy
        self.by -= learning_rate * dby
        
        return dh_prev, dc_prev


# ==================== 4. GRU 单元 ====================
class GRUCell:
    """GRU (Gated Recurrent Unit) 单元"""
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化 GRU 单元
        
        参数:
            input_size: 输入维度
            hidden_size: 隐藏层维度
            output_size: 输出维度
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 重置门权重
        self.Wr = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.br = np.zeros((1, hidden_size))
        
        # 更新门权重
        self.Wz = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.bz = np.zeros((1, hidden_size))
        
        # 候选隐藏状态权重
        self.Wh = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.bh = np.zeros((1, hidden_size))
        
        # 输出层权重
        self.Wy = np.random.randn(hidden_size, output_size) * 0.01
        self.by = np.zeros((1, output_size))
        
        self.cache = {}
    
    def forward(self, x, h_prev):
        """
        GRU 前向传播
        
        参数:
            x: 输入
            h_prev: 上一时刻隐藏状态
        返回:
            h_next: 当前隐藏状态
            y: 输出
        """
        # 拼接输入和隐藏状态
        concat = np.concatenate([x, h_prev], axis=1)
        
        # 重置门: r_t = σ(W_r · [h_{t-1}, x_t] + b_r)
        r = Activation.sigmoid(np.dot(concat, self.Wr) + self.br)
        
        # 更新门: z_t = σ(W_z · [h_{t-1}, x_t] + b_z)
        z = Activation.sigmoid(np.dot(concat, self.Wz) + self.bz)
        
        # 候选隐藏状态: h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)
        concat_reset = np.concatenate([x, r * h_prev], axis=1)
        h_tilde = np.tanh(np.dot(concat_reset, self.Wh) + self.bh)
        
        # 隐藏状态: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
        h_next = (1 - z) * h_prev + z * h_tilde
        
        # 输出
        y = np.dot(h_next, self.Wy) + self.by
        
        # 保存中间值
        self.cache = {
            'x': x, 'h_prev': h_prev, 'concat': concat,
            'r': r, 'z': z, 'concat_reset': concat_reset,
            'h_tilde': h_tilde, 'h_next': h_next, 'y': y
        }
        
        return h_next, y
    
    def backward(self, dy, dh_next, learning_rate=0.01):
        """
        GRU 反向传播
        
        参数:
            dy: 输出梯度
            dh_next: 下一时刻隐藏状态梯度
            learning_rate: 学习率
        返回:
            dh_prev: 上一时刻隐藏状态梯度
        """
        # 提取缓存值
        x = self.cache['x']
        h_prev = self.cache['h_prev']
        concat = self.cache['concat']
        r = self.cache['r']
        z = self.cache['z']
        concat_reset = self.cache['concat_reset']
        h_tilde = self.cache['h_tilde']
        h_next = self.cache['h_next']
        
        # 输出层梯度
        dWy = np.dot(h_next.T, dy)
        dby = np.sum(dy, axis=0, keepdims=True)
        
        # 隐藏状态梯度
        dh = np.dot(dy, self.Wy.T) + dh_next
        
        # 更新门梯度
        dz = dh * (h_tilde - h_prev)
        dz_raw = dz * z * (1 - z)
        
        # 候选隐藏状态梯度
        dh_tilde = dh * z
        dh_tilde_raw = dh_tilde * (1 - h_tilde ** 2)
        
        # 重置门梯度
        dconcat_reset = np.dot(dh_tilde_raw, self.Wh.T)
        dr = dconcat_reset[:, self.input_size:] * h_prev
        dr_raw = dr * r * (1 - r)
        
        # 权重梯度
        dWz = np.dot(concat.T, dz_raw)
        dbz = np.sum(dz_raw, axis=0, keepdims=True)
        
        dWr = np.dot(concat.T, dr_raw)
        dbr = np.sum(dr_raw, axis=0, keepdims=True)
        
        dWh = np.dot(concat_reset.T, dh_tilde_raw)
        dbh = np.sum(dh_tilde_raw, axis=0, keepdims=True)
        
        # 上一时刻隐藏状态梯度
        dh_prev = (dh * (1 - z) + 
                   dconcat_reset[:, self.input_size:] * r +
                   np.dot(dz_raw, self.Wz.T)[:, self.input_size:] +
                   np.dot(dr_raw, self.Wr.T)[:, self.input_size:])
        
        # 梯度裁剪
        for grad in [dWr, dWz, dWh, dWy]:
            np.clip(grad, -5, 5, out=grad)
        
        # 更新权重
        self.Wr -= learning_rate * dWr
        self.br -= learning_rate * dbr
        self.Wz -= learning_rate * dWz
        self.bz -= learning_rate * dbz
        self.Wh -= learning_rate * dWh
        self.bh -= learning_rate * dbh
        self.Wy -= learning_rate * dWy
        self.by -= learning_rate * dby
        
        return dh_prev


# ==================== 5. RNN 模型 ====================
class RNN:
    """RNN 模型（支持 Vanilla RNN, LSTM, GRU）"""
    
    def __init__(self, input_size, hidden_size, output_size, cell_type='lstm'):
        """
        初始化 RNN 模型
        
        参数:
            input_size: 输入维度
            hidden_size: 隐藏层维度
            output_size: 输出维度
            cell_type: 'vanilla', 'lstm', 'gru'
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cell_type = cell_type
        
        # 创建 RNN 单元
        if cell_type == 'vanilla':
            self.cell = VanillaRNNCell(input_size, hidden_size, output_size)
        elif cell_type == 'lstm':
            self.cell = LSTMCell(input_size, hidden_size, output_size)
        elif cell_type == 'gru':
            self.cell = GRUCell(input_size, hidden_size, output_size)
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")
        
        self.history = {'loss': [], 'accuracy': []}
    
    def forward(self, X, h_init=None, c_init=None):
        """
        前向传播整个序列
        
        参数:
            X: 输入序列 (seq_len, batch_size, input_size)
            h_init: 初始隐藏状态
            c_init: 初始细胞状态（仅LSTM使用）
        返回:
            outputs: 所有时刻的输出
            hiddens: 所有时刻的隐藏状态
        """
        seq_len, batch_size, _ = X.shape
        
        # 初始化隐藏状态
        if h_init is None:
            h = np.zeros((batch_size, self.hidden_size))
        else:
            h = h_init
        
        if self.cell_type == 'lstm':
            if c_init is None:
                c = np.zeros((batch_size, self.hidden_size))
            else:
                c = c_init
        
        # 保存所有时刻的输出和隐藏状态
        outputs = []
        hiddens = []
        
        # 遍历序列
        for t in range(seq_len):
            if self.cell_type == 'vanilla' or self.cell_type == 'gru':
                h, y = self.cell.forward(X[t], h)
            elif self.cell_type == 'lstm':
                h, c, y = self.cell.forward(X[t], h, c)
            
            outputs.append(y)
            hiddens.append(h)
        
        return np.array(outputs), np.array(hiddens)
    
    def backward(self, X, targets, outputs, learning_rate=0.01):
        """
        反向传播
        
        参数:
            X: 输入序列
            targets: 目标序列
            outputs: 前向传播的输出
            learning_rate: 学习率
        返回:
            loss: 损失值
        """
        seq_len, batch_size, _ = X.shape
        
        # 计算损失
        loss = 0
        dh_next = np.zeros((batch_size, self.hidden_size))
        
        if self.cell_type == 'lstm':
            dc_next = np.zeros((batch_size, self.hidden_size))
        
        # 反向传播通过时间（BPTT）
        for t in reversed(range(seq_len)):
            # Softmax + 交叉熵损失
            probs = Activation.softmax(outputs[t])
            
            # 计算损失
            loss += -np.mean(np.log(probs[range(batch_size), targets[t]] + 1e-10))
            
            # 输出梯度
            dy = probs.copy()
            dy[range(batch_size), targets[t]] -= 1
            dy /= batch_size
            
            # 反向传播
            if self.cell_type == 'vanilla' or self.cell_type == 'gru':
                dh_next = self.cell.backward(dy, dh_next, learning_rate)
            elif self.cell_type == 'lstm':
                dh_next, dc_next = self.cell.backward(dy, dh_next, dc_next, learning_rate)
        
        return loss / seq_len
    
    def train_step(self, X, targets, learning_rate=0.01):
        """训练一步"""
        # 前向传播
        outputs, _ = self.forward(X)
        
        # 反向传播
        loss = self.backward(X, targets, outputs, learning_rate)
        
        # 计算准确率
        predictions = np.argmax(Activation.softmax(outputs), axis=2)
        accuracy = np.mean(predictions == targets)
        
        return loss, accuracy
    
    def predict(self, X):
        """预测"""
        outputs, _ = self.forward(X)
        predictions = np.argmax(Activation.softmax(outputs), axis=2)
        return predictions


# ==================== 6. 生成序列数据 ====================
print("\n[步骤 1] 生成序列数据...")

def generate_sequence_data(num_samples=1000, seq_len=10):
    """
    生成简单的序列数据：计数任务
    输入: [1, 2, 3, 4, 5, ...]
    输出: [2, 3, 4, 5, 6, ...] (下一个数字)
    """
    X = []
    y = []
    
    for _ in range(num_samples):
        start = np.random.randint(0, 50)
        sequence = np.arange(start, start + seq_len + 1)
        
        X.append(sequence[:-1])
        y.append(sequence[1:])
    
    return np.array(X), np.array(y)

# 生成数据
num_samples = 500
seq_len = 8
vocab_size = 60  # 0-59

X_train, y_train = generate_sequence_data(num_samples, seq_len)
X_test, y_test = generate_sequence_data(100, seq_len)

print(f"训练集: {X_train.shape}, 标签: {y_train.shape}")
print(f"测试集: {X_test.shape}, 标签: {y_test.shape}")
print(f"\n示例序列:")
print(f"  输入: {X_train[0]}")
print(f"  输出: {y_train[0]}")


# ==================== 7. One-hot 编码 ====================
def to_one_hot(sequences, vocab_size):
    """将序列转换为 one-hot 编码"""
    num_samples, seq_len = sequences.shape
    one_hot = np.zeros((num_samples, seq_len, vocab_size))
    
    for i in range(num_samples):
        for t in range(seq_len):
            one_hot[i, t, sequences[i, t]] = 1
    
    return one_hot

X_train_onehot = to_one_hot(X_train, vocab_size)
X_test_onehot = to_one_hot(X_test, vocab_size)

# 转换维度: (batch, seq, features) -> (seq, batch, features)
X_train_onehot = np.transpose(X_train_onehot, (1, 0, 2))
X_test_onehot = np.transpose(X_test_onehot, (1, 0, 2))
y_train = y_train.T
y_test = y_test.T

print(f"\nOne-hot 编码后:")
print(f"  X_train: {X_train_onehot.shape}")
print(f"  X_test: {X_test_onehot.shape}")


# ==================== 8. 训练模型 ====================
print("\n[步骤 2] 训练 RNN 模型...")
print("=" * 70)

# 比较三种 RNN 类型
cell_types = ['vanilla', 'lstm', 'gru']
models = {}
histories = {}

for cell_type in cell_types:
    print(f"\n训练 {cell_type.upper()} 模型...")
    print("-" * 70)
    
    # 创建模型
    model = RNN(
        input_size=vocab_size,
        hidden_size=32,
        output_size=vocab_size,
        cell_type=cell_type
    )
    
    # 训练参数
    epochs = 10
    batch_size = 32
    learning_rate = 0.01
    
    # 训练循环
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        num_batches = X_train_onehot.shape[1] // batch_size
        
        for batch in range(num_batches):
            # 获取批次数据
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            
            X_batch = X_train_onehot[:, start_idx:end_idx, :]
            y_batch = y_train[:, start_idx:end_idx]
            
            # 训练一步
            loss, acc = model.train_step(X_batch, y_batch, learning_rate)
            
            epoch_loss += loss
            epoch_acc += acc
        
        # 计算平均值
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        
        # 保存历史
        model.history['loss'].append(avg_loss)
        model.history['accuracy'].append(avg_acc)
        
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} - "
                  f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
    
    # 在测试集上评估
    test_predictions = model.predict(X_test_onehot)
    test_acc = np.mean(test_predictions == y_test)
    
    print(f"  测试准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    models[cell_type] = model
    histories[cell_type] = model.history

print("\n" + "=" * 70)
print("✓ 训练完成！")


# ==================== 9. 可视化比较 ====================
print("\n[步骤 3] 可视化结果...")

# 训练历史比较
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = {'vanilla': 'blue', 'lstm': 'green', 'gru': 'red'}
for cell_type in cell_types:
    history = histories[cell_type]
    
    axes[0].plot(history['loss'], label=cell_type.upper(), 
                 color=colors[cell_type], linewidth=2, marker='o')
    axes[1].plot(history['accuracy'], label=cell_type.upper(),
                 color=colors[cell_type], linewidth=2, marker='o')

axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('训练损失对比', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('训练准确率对比', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rnn_comparison.png', dpi=150, bbox_inches='tight')
print("✓ 训练对比图已保存: rnn_comparison.png")


# ==================== 10. 预测示例 ====================
print("\n[步骤 4] 预测示例...")

# 选择 LSTM 模型进行预测
best_model = models['lstm']

# 预测几个测试样本
num_examples = 5
for i in range(num_examples):
    X_sample = X_test_onehot[:, i:i+1, :]
    y_true = y_test[:, i]
    
    prediction = best_model.predict(X_sample)
    prediction = prediction.flatten()
    
    # 原始输入序列
    input_seq = X_test[i]
    
    print(f"\n示例 {i+1}:")
    print(f"  输入序列: {input_seq}")
    print(f"  真实输出: {y_true}")
    print(f"  预测输出: {prediction}")
    print(f"  正确: {'✓' if np.array_equal(prediction, y_true) else '✗'}")


# ==================== 11. 可视化预测 ====================
# 可视化序列预测
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

for idx, cell_type in enumerate(cell_types):
    model = models[cell_type]
    
    # 预测测试集
    predictions = model.predict(X_test_onehot)
    
    # 显示前50个时间步的预测
    display_len = min(50, predictions.shape[1])
    
    ax = axes[idx]
    
    # 绘制真实值
    true_flat = y_test[:, :display_len].flatten()
    pred_flat = predictions[:, :display_len].flatten()
    
    x_axis = np.arange(len(true_flat))
    
    ax.plot(x_axis, true_flat, 'b-', linewidth=2, label='真实值', alpha=0.7)
    ax.plot(x_axis, pred_flat, 'r--', linewidth=2, label='预测值', alpha=0.7)
    
    # 标记错误预测
    errors = true_flat != pred_flat
    if np.any(errors):
        ax.scatter(x_axis[errors], pred_flat[errors], 
                  color='red', s=100, marker='x', linewidths=3, 
                  label='错误预测', zorder=5)
    
    accuracy = np.mean(predictions == y_test)
    ax.set_title(f'{cell_type.upper()} - 准确率: {accuracy:.2%}', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('时间步', fontsize=11)
    ax.set_ylabel('值', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rnn_predictions.png', dpi=150, bbox_inches='tight')
print("\n✓ 预测结果已保存: rnn_predictions.png")


# ==================== 12. 性能统计 ====================
print("\n[步骤 5] 性能统计...")
print("=" * 70)

results = []
for cell_type in cell_types:
    model = models[cell_type]
    
    # 测试集预测
    predictions = model.predict(X_test_onehot)
    accuracy = np.mean(predictions == y_test)
    
    # 最终损失
    final_loss = model.history['loss'][-1]
    
    results.append({
        'model': cell_type.upper(),
        'accuracy': accuracy,
        'final_loss': final_loss
    })
    
    print(f"{cell_type.upper():10s} - 准确率: {accuracy:.4f} ({accuracy*100:.2f}%), "
          f"最终损失: {final_loss:.4f}")

print("=" * 70)


# ==================== 13. 模型架构可视化 ====================
print("\n[步骤 6] 模型架构说明...")

architectures = {
    'Vanilla RNN': """
    输入 (x_t) → [线性变换] → tanh → 隐藏状态 (h_t)
                   ↑
              上一隐藏状态 (h_{t-1})
    
    公式: h_t = tanh(W_xh·x_t + W_hh·h_{t-1} + b_h)
          y_t = W_hy·h_t + b_y
    
    优点: 简单、计算快
    缺点: 梯度消失/爆炸
    """,
    
    'LSTM': """
    输入 (x_t) + 上一隐藏状态 (h_{t-1})
         ↓
    [遗忘门 f_t] → 决定遗忘多少信息
    [输入门 i_t] → 决定更新多少信息
    [候选值 c̃_t] → 新的候选信息
    [输出门 o_t] → 决定输出多少信息
         ↓
    细胞状态 (c_t) = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
    隐藏状态 (h_t) = o_t ⊙ tanh(c_t)
    
    优点: 解决长期依赖问题
    缺点: 参数多、计算慢
    """,
    
    'GRU': """
    输入 (x_t) + 上一隐藏状态 (h_{t-1})
         ↓
    [重置门 r_t] → 决定保留多少历史信息
    [更新门 z_t] → 决定更新多少信息
    [候选状态 h̃_t] → 新的候选隐藏状态
         ↓
    隐藏状态 (h_t) = (1-z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
    
    优点: 比LSTM参数少、效果相近
    缺点: 比Vanilla RNN复杂
    """
}

for name, arch in architectures.items():
    print(f"\n{name}:")
    print(arch)


# ==================== 14. 文本生成示例 ====================
print("\n[步骤 7] 文本生成示例...")

def generate_sequence(model, start_seq, length=20, temperature=1.0):
    """
    生成新序列
    
    参数:
        model: 训练好的模型
        start_seq: 起始序列
        length: 生成长度
        temperature: 温度参数（控制随机性）
    返回:
        生成的序列
    """
    generated = list(start_seq)
    current_seq = start_seq.copy()
    
    for _ in range(length):
        # 转换为 one-hot
        seq_onehot = to_one_hot(current_seq.reshape(1, -1), vocab_size)
        seq_onehot = np.transpose(seq_onehot, (1, 0, 2))
        
        # 预测下一个值
        outputs, _ = model.forward(seq_onehot)
        logits = outputs[-1, 0, :]  # 最后一个时间步的输出
        
        # 应用温度
        logits = logits / temperature
        probs = Activation.softmax(logits.reshape(1, -1)).flatten()
        
        # 采样
        next_val = np.random.choice(vocab_size, p=probs)
        
        generated.append(next_val)
        
        # 更新当前序列（滑动窗口）
        current_seq = np.append(current_seq[1:], next_val)
    
    return np.array(generated)

# 生成示例
start_sequence = np.array([10, 11, 12, 13, 14, 15, 16, 17])
print(f"\n起始序列: {start_sequence}")

for cell_type in cell_types:
    model = models[cell_type]
    generated = generate_sequence(model, start_sequence, length=12, temperature=0.5)
    print(f"{cell_type.upper():10s} 生成: {generated}")

expected = np.arange(10, 10 + len(start_sequence) + 12)
print(f"{'预期':10s} 序列: {expected}")


# ==================== 15. 总结报告 ====================
print("\n" + "=" * 70)
print("项目总结报告")
print("=" * 70)
print(f"任务: 序列预测（计数任务）")
print(f"训练样本: {num_samples}")
print(f"测试样本: {100}")
print(f"序列长度: {seq_len}")
print(f"词汇表大小: {vocab_size}")
print(f"隐藏层大小: 32")
print(f"训练轮次: {epochs}")
print(f"\n实现的模型:")
print(f"  ✓ Vanilla RNN - 基础循环神经网络")
print(f"  ✓ LSTM - 长短期记忆网络")
print(f"  ✓ GRU - 门控循环单元")
print(f"\n实现的组件:")
print(f"  ✓ 前向传播 (Forward Propagation)")
print(f"  ✓ 反向传播 (Backpropagation Through Time)")
print(f"  ✓ 梯度裁剪 (Gradient Clipping)")
print(f"  ✓ 激活函数 (tanh, sigmoid, softmax, ReLU)")
print(f"  ✓ 损失函数 (Cross Entropy Loss)")
print(f"\n模型性能:")
for result in results:
    print(f"  {result['model']:10s}: 准确率 {result['accuracy']:.2%}, "
          f"损失 {result['final_loss']:.4f}")
print(f"\n生成文件:")
print(f"  ✓ rnn_comparison.png - 模型训练对比")
print(f"  ✓ rnn_predictions.png - 预测结果可视化")
print("=" * 70)
print("\n✓ 完成！纯 NumPy RNN/LSTM/GRU 实现成功运行。")
print("\n关键发现:")
print("  - LSTM 和 GRU 在长序列上表现更好")
print("  - Vanilla RNN 容易出现梯度消失")
print("  - GRU 参数更少，训练更快")
print("  - LSTM 在复杂任务上通常效果最好")
print("=" * 70)


# ==================== 16. 额外：保存和加载模型 ====================
print("\n[额外功能] 模型保存示例...")

def save_model(model, filename):
    """保存模型参数"""
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ 模型已保存: {filename}")

def load_model(filename):
    """加载模型参数"""
    import pickle
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ 模型已加载: {filename}")
    return model

# 保存最佳模型
save_model(models['lstm'], 'lstm_model.pkl')

print("\n" + "=" * 70)
print("全部完成！✓")
print("=" * 70)