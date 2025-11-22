#TRAIN.PY
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------
# 1) DATA LOADING
# -----------------------
DATASET_PATH = "brain_mri"
IMG_SIZE = 150
data = []
labels = []

# Folder names represent labels
for label in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, label)
    
    if not os.path.isdir(folder_path):
        continue

    for img_name in os.listdir(folder_path):
        try:
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(1 if label == "yes" else 0)
        except:
            pass

print("Dataset Loaded: ", len(data), "images")

# Convert to numpy
data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
labels = np.array(labels)

# Normalize
data = data / 255.0

# -----------------------
# 2) SPLIT DATA
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Convert labels to categorical (One-hot encoding)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# -----------------------
# 3) DATA AUGMENTATION (optional)
# -----------------------
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# -----------------------
# 4) MODEL BUILDING
# -----------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------
# 5) TRAINING
# -----------------------
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=20
)

# -----------------------
# 6) MODEL EVALUATION
# -----------------------
loss, acc = model.evaluate(X_test, y_test)
print(f"\nModel Accuracy: {acc * 100:.2f}%")

# -----------------------
# 7) SAVE MODEL
# -----------------------
model.save("brain_mri_classifier.h5")
print("Model Saved as brain_mri_classifier.h5")

# -----------------------
# 8) PLOT RESULTS
# -----------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss")

plt.show()
