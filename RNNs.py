#tensorflow to reliaze the differences between RNN, LSTM, and GRU
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense
import numpy as np

# ====== 1️⃣ 生成示例数据 ======
# 样本数 = 500，时间步 = 20，每步特征 = 5
X = np.random.random((500, 20, 5))
# 二分类标签
y = np.random.randint(0, 2, size=(500, 1))

# ====== 2️⃣ 定义三种模型结构 ======

# --- 普通 RNN ---
rnn_model = Sequential([
    SimpleRNN(64, input_shape=(20, 5)),
    Dense(1, activation='sigmoid')
])
rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- LSTM ---
lstm_model = Sequential([
    LSTM(64, input_shape=(20, 5)),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- GRU ---
gru_model = Sequential([
    GRU(64, input_shape=(20, 5)),
    Dense(1, activation='sigmoid')
])
gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ====== 3️⃣ 分别训练三个模型 ======
print("\n🧠 Training Simple RNN...")
rnn_model.fit(X, y, epochs=3, batch_size=32, verbose=1)

print("\n🧠 Training LSTM...")
lstm_model.fit(X, y, epochs=3, batch_size=32, verbose=1)

print("\n🧠 Training GRU...")
gru_model.fit(X, y, epochs=3, batch_size=32, verbose=1)

# ====== 4️⃣ 对比结果 ======
print("\n✅ RNN Evaluation:")
print(rnn_model.evaluate(X, y))

print("\n✅ LSTM Evaluation:")
print(lstm_model.evaluate(X, y))

print("\n✅ GRU Evaluation:")
print(gru_model.evaluate(X, y))
