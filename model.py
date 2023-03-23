import zipfile
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.utils import to_categorical
import pandas as pd



from google.colab import drive
drive.mount('/content/drive')

with zipfile.ZipFile("/content/drive/MyDrive/Gasleaksdata.zip", 'r') as zip_ref:
    zip_ref.extractall()
data_dir = 'Gasleaksdata/'

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

def preprocess_image(image):    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    return dilated.reshape((dilated.shape[0], dilated.shape[1], 1))

def identify_gas(image):
    mean = np.mean(image)
    # If the mean pixel value is less than 80, the gas is atmospheric gas (label = 0)
    if mean < 80:
        return 0
    # If the mean pixel value is between 80 and 120, the gas is cold (label = 1)
    elif mean >= 80 and mean < 120:
        return 1
    # If the mean pixel value is between 120 and 160, the gas is moderate temp (label = 2)
    elif mean >= 120 and mean < 160:
        return 2
    # If the mean pixel value is greater than or equal to 160, the gas is hot temp (label = 3)
    else:
        return 3

images = []
labels = []
for file in os.listdir(data_dir):
    if file.endswith('.png'):
        image = cv2.imread(os.path.join(data_dir, file))
        images.append(preprocess_image(image))
        labels.append(identify_gas(image))

images = np.array(images)
labels = np.array(labels)
labels = to_categorical(labels,num_classes=4)

import random
random.seed(42)  

X_train, X_test, y_train, y_testm = train_test_split(images, labels, test_size=0.2,random_state=42)

with tpu_strategy.scope():
  modelmn = models.Sequential()
  modelmn.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
  modelmn.add(layers.MaxPooling2D((2, 2)))
  modelmn.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
  modelmn.add(layers.MaxPooling2D((2, 2)))
  modelmn.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
  modelmn.add(layers.Flatten())
  modelmn.add(layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
  modelmn.add(layers.Dense(4, activation='softmax'))

  modelmn.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
  modelmn.fit(X_train, y_train, epochs=100 ,validation_data=(X_test, y_testm))
  y_predm = np.argmax(modelmn.predict(X_test), axis=1)
  accuracy = accuracy_score(np.argmax(y_testm, axis=1), y_predm)
  print(accuracy*100)
  conf_matrix = confusion_matrix(np.argmax(y_testm, axis=1), y_predm)
  print("Confusion matrix:\n", conf_matrix)