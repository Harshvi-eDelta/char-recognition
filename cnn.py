import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 
import cv2
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

digits = pd.read_csv("/Users/edelta076/Desktop/char_recognition/archive 2/mnist_train.csv")
print(digits)   
print(digits.shape)
print(digits.head())

x = digits.drop(columns=['label'])  
y = digits['label'] 

x1 = x.values.reshape(-1,28, 28, 1) / 255.0  
plt.imshow(x1[6], cmap='gray')
plt.title(f"Image Label: {y[6]}")
#plt.show()

x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size=0.3, random_state=42)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  
])

model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=6)

loss, ac = model.evaluate(x_test, y_test)
print("Loss:", loss)
print("Accuracy:", ac)

predictions = model.predict(x_test)
y_pred = np.argmax(predictions, axis=1)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

img_num = 0

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))  
    img = img / 255.0 
    img = img.reshape(1, 28,28, 1)  
    return img

while os.path.isfile(f"/Users/edelta076/Desktop/char_recognition/img{img_num}.png") :
    image_path = f"/Users/edelta076/Desktop/char_recognition/img{img_num}.png"
    img = preprocess_image(image_path)
    pred = model.predict(img)
    print(f"Prediction for image {img_num} is: {np.argmax(pred)}")
    img_to_show = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img_to_show,cmap='gray')
    plt.title(f"Predicted: {np.argmax(pred)}")
    plt.show()
    img_num += 1

'''image_path = "/Users/edelta076/Desktop/char_recognition/img4.png"
img = preprocess_image(image_path)
prediction = model.predict(img)
print(f"Predicted digit: {np.argmax(prediction)}")

img_to_show = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
plt.imshow(img_to_show, cmap='gray')
plt.title(f"Predicted: {np.argmax(prediction)}")
plt.show()'''
