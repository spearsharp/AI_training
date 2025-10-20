import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# ====== 1️⃣ 加载 MNIST 数据集 ======
# 包含 60,000 张训练图像 和 10,000 张测试图像，灰度手写数字(0-9)
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# ====== 2️⃣ 数据预处理 ======
# CNN 需要输入 4D 张量 (batch, height, width, channels)
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images  = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# ====== 3️⃣ 构建 CNN 模型 ======
model = models.Sequential([
    # 卷积层 1 + 池化层 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    # 卷积层 2 + 池化层 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # 卷积层 3
    layers.Conv2D(64, (3, 3), activation='relu'),

    # 平铺并进入全连接层
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 输出 10 类
])

# ====== 4️⃣ 编译模型 ======
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ====== 5️⃣ 训练模型 ======
history = model.fit(train_images, train_labels, epochs=5, 
                    batch_size=64,
                    validation_data=(test_images, test_labels))

# ====== 6️⃣ 测试模型性能 ======
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\n✅ 测试集准确率: {test_acc:.4f}")

# ====== 7️⃣ 可视化训练过程 ======
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('CNN Accuracy Curve')
plt.show()
