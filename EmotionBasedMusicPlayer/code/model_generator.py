import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import cv2, numpy as np
import keras
from keras import *
import os
import pandas as pd
import matplotlib.pyplot as plt

def load_images_and_labels(data_path):
  images = []
  labels = []
  for emotion in os.listdir(data_path):
    emotion_path = os.path.join(data_path, emotion)
    for filename in os.listdir(emotion_path):
      image_path = os.path.join(emotion_path, filename)
      img = cv2.imread(image_path)
      images.append(img.flatten())
      labels.append(emotion)
  return np.array(images), np.array(labels)

def preprocess_image(img, target_size=(48, 48)):
  img = cv2.resize(img, target_size)
  img = img.astype('float32') / 255.0
  img = img.reshape(target_size[0], target_size[1], 1)
  return img

data_path = "code/Dataset/train"
num_emotions = 7
images, labels = load_images_and_labels(data_path)

preprocessed_images = np.array([preprocess_image(img) for img in images])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
label_numbers = le.fit_transform(labels)

from tensorflow.keras.utils import to_categorical
labels_onehot = to_categorical(label_numbers, num_classes=num_emotions)

x_train, x_test_val, y_train, y_test_val = train_test_split(preprocessed_images, labels_onehot, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test_val, y_test_val, test_size=0.5, random_state=42)  # Further split for validation and test

# Define the CNN model
model = keras.Sequential([
  layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(x_train.shape[1], x_train.shape[2], 1)),
  layers.MaxPooling2D(pool_size=(2, 2)),
  
  layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
  layers.MaxPooling2D(pool_size=(2, 2)),
  
  layers.Flatten(),
  
  layers.Dense(128, activation='relu'),
  
  layers.Dense(num_emotions, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

model.save('emotion_recognition_model.h5')

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
plt.show()