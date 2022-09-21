# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images the dataset contains 70000 samples which is generated from tensorflow

## Neural Network Model

![image](https://user-images.githubusercontent.com/75235167/191184632-d7864b39-06b7-48ff-8c0c-c4092d8596f5.png)

## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries

### STEP 2:
Build a CNN model

### STEP 3:
Compile and fit the model and then predict

## PROGRAM
```python3
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

np.unique(y_test)

model = keras.Sequential([
    tf.keras.layers.Conv2D(32,kernel_size=3,activation="relu",padding="same"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(16,kernel_size=3,activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(8,kernel_size=3,activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10,activation="softmax")
])

model.compile(loss="categorical_crossentropy", metrics='accuracy',optimizer="adam")

model.fit(X_train_scaled ,y_train_onehot, epochs=2,
          batch_size=64, 
          validation_data=(X_test_scaled,y_test_onehot))

pd.DataFrame(model.history.history).plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

confusion_matrix(y_test,x_test_predictions)

print(classification_report(y_test,x_test_predictions))

img = image.load_img('imageeight.jpg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

np.argmax(model.predict(img_28_gray_scaled.reshape(1,28,28,1)),axis=1)
```
## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot

![image](https://user-images.githubusercontent.com/75235167/191184943-beaeb8bd-ad6d-475f-be79-360efa4f8b79.png)

### Classification Report

![image](https://user-images.githubusercontent.com/75235167/191185022-499021a3-67aa-4202-bd1f-3054e599a083.png)

### Confusion Matrix

![image](https://user-images.githubusercontent.com/75235167/191185099-bc0f33a2-b6a6-4f1d-afe7-e2c348f779e5.png)

### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/75235167/191185232-9d299919-7280-47b9-b667-429ede897d59.png)

## RESULT

Thus a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
