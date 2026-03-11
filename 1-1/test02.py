import tensorflow as tf
from tensorflow.keras import layers, models

# 1. 데이터셋 불러오기
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "~/dataset/",
    image_size=(128, 128),
    batch_size=32
)

# 2. 데이터 전처리 (정규화)
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# 3. 모델 정의 (간단 CNN)
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')  # cats vs dogs
])

# 4. 컴파일 & 학습
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, epochs=5)

# 5. 모델 저장
model.save("saved_model/catdog_model.h5")

# # 6. 새로운 이미지 예측
# from PIL import Image
# import numpy as np

# img = Image.open("test_cat.jpg").resize((128,128))
# img_array = np.array(img).astype("float32")/255.0
# img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가

# prediction = model.predict(img_array)
# print("예측 결과:", prediction)
# print("클래스:", np.argmax(prediction))  # 0=cat, 1=dog


