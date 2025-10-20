import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# 模拟输入数据：样本数=100，时间步=10，每步3个特征
X = np.random.rand(100, 10, 3)
# 模拟输出数据：二分类
y = np.random.randint(0, 2, size=(100, 1))

# 构建 LSTM 模型
model = Sequential([
    LSTM(64, input_shape=(10, 3)),  # LSTM层替换RNN
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
model.fit(X, y, epochs=5, batch_size=16)
