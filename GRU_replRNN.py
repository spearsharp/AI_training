import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import numpy as np

# 模拟输入数据：样本数=100，时间步=10，每步3个特征
X = np.random.rand(100, 10, 3)
y = np.random.randint(0, 2, size=(100, 1))

# 构建 GRU 模型
model = Sequential([
    GRU(64, input_shape=(10, 3)),  # GRU层替换RNN
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
model.fit(X, y, epochs=5, batch_size=16)
