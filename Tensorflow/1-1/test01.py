import tensorflow as tf
from tensorflow.keras import layers, models

# 1. 데이터셋 불러오기 (MNIST: 0~9 손글씨 이미지)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. 데이터 전처리 (정규화 + 차원 맞추기)
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# 3. 간단한 CNN 모델 정의
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 0~9 숫자 분류
])

# 4. 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. 학습
model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.1)

# 6. 모델 저장
model.save("saved_model/number_model.h5")

# 7. 평가
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("테스트 정확도:", test_acc)
