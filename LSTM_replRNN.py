import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Dense
import numpy as np

# 模拟输入数据：样本数=1000，时间步=10，每步3个特征
X = np.random.rand(1000, 10, 3)
# 模拟输出数据：二分类（与X样本数量一致）
y = np.random.randint(0, 2, size=(1000, 1))

# 构建 LSTM 模型 (修复警告: 使用Input层替代input_shape)
model = keras.Sequential([
    Input(shape=(10, 3)),           # 明确的输入层
    LSTM(64),                       # LSTM层替换RNN
    Dense(1, activation='sigmoid')  # 输出层
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("📋 模型架构:")
model.summary()

print("\n🚀 开始训练 LSTM 模型...")
history = model.fit(X, y, epochs=5, batch_size=16, validation_split=0.2, verbose=1)

print("\n✅ 训练完成!")
print(f"最终训练精度: {history.history['accuracy'][-1]:.4f}")
print(f"最终验证精度: {history.history['val_accuracy'][-1]:.4f}")

# 进行预测
print("\n🔮 进行预测测试...")
test_sample = X[:5]  # 取前5个样本进行测试
predictions = model.predict(test_sample, verbose=0)
print("预测结果 (概率):", predictions.flatten())
print("实际标签:", y[:5].flatten())
print("预测类别:", (predictions > 0.5).astype(int).flatten())
