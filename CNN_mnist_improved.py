import numpy as np
from tensorflow.keras.datasets import mnist

# ====== Improved CNN Implementation with Full Training ======

class Conv3x3:
    """3x3 Convolution Layer with backpropagation"""
    def __init__(self, num_filters):
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, 3, 3) * 0.1
        self.last_input = None
    
    def forward(self, input_img):
        self.last_input = input_img
        h, w = input_img.shape
        self.output_shape = (self.num_filters, h - 2, w - 2)
        output = np.zeros(self.output_shape)
        
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

# 使用 TensorFlow/Keras 进行实际训练（更高效）
import tensorflow as tf
from tensorflow.keras import layers, models

def create_improved_cnn():
    """Create an improved CNN model using Keras"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 加载 MNIST 数据
print("🔄 Loading MNIST dataset...")
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 归一化数据
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# 重塑数据以适应CNN (添加通道维度)
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# 使用更多数据进行训练
N_train = 10000  # 增加到10k样本
N_test = 2000    # 增加到2k样本

X_train = train_images[:N_train]
y_train = train_labels[:N_train]
X_test = test_images[:N_test]
y_test = test_labels[:N_test]

print(f"📊 Training data shape: {X_train.shape}")
print(f"📊 Test data shape: {X_test.shape}")

# 创建并训练改进的CNN模型
print("\n🧠 Creating improved CNN model...")
model = create_improved_cnn()

# 显示模型结构
print("\n📋 Model Architecture:")
model.summary()

# 训练模型
print("\n🚀 Training the model...")
history = model.fit(X_train, y_train,
                    epochs=5,
                    batch_size=128,
                    validation_split=0.2,
                    verbose=1)

# 评估模型
print("\n📊 Evaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test accuracy: {test_acc:.4f}")

# 显示一些预测结果
print("\n🔍 Sample predictions:")
predictions = model.predict(X_test[:10], verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

for i in range(10):
    print(f"Image {i}: True={y_test[i]}, Predicted={predicted_labels[i]}, "
          f"Confidence={predictions[i][predicted_labels[i]]:.3f}")

print("\n" + "="*60)
print("🎉 CNN MNIST Training Complete!")
print(f"📈 Final Test Accuracy: {test_acc:.2%}")
print("="*60)