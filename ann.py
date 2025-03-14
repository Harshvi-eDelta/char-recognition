import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 
import cv2
import os
from sklearn.metrics import accuracy_score
import pytesseract
import pandas as pd
from sklearn.model_selection import train_test_split

#mnist = tf.keras.datasets.mnist

#(x_train,y_train),(x_test,y_test) = mnist.load_data()

#x_train = tf.keras.utils.normalize(x_train,axis=1) 
#x_test = tf.keras.utils.normalize(x_test,axis=1)

digits = pd.read_csv("/Users/edelta076/Desktop/char_recognition/archive 2/mnist_train.csv")
print(digits.shape)
print(digits.head())

x = digits.drop(columns=['label'])
y = digits['label']
print(x)

x1 = x.values.reshape(-1,28,28,1)
plt.imshow(x1[6],cmap='gray')
print("image is :" , y[6])

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.3)
x_train = x_train.values.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.values.reshape(-1, 28, 28, 1) / 255.0


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28,1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])
model.fit(x_train,y_train,epochs=3)
loss,ac = model.evaluate(x_test,y_test)
print("loss" ,loss)
print("accuracy score :",ac)

img_num = 0

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
    img = cv2.resize(img, (28, 28))  
    if np.mean(img) > 125 :
        img = 255 - img
    img = img / 255.0  
    #img = img.reshape(1, -1)  
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1)
    return img

'''def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
    img = cv2.resize(img,(28,28))  
    if np.mean(img) > 127 :
        img = 255-img
    img = img / 255.0 
    img = np.expand_dims(img, axis=0)  
    img = np.expand_dims(img, axis=-1) 
    return img'''

'''image_path = f"/Users/edelta076/Desktop/char_recognition/img3.png"

img = preprocess_image(image_path)
pred = model.predict(img)
print(pred)
prediction = np.argmax(pred)
print(f"Prediction for image is: {prediction}")

plt.imshow(img[0],cmap=plt.cm.binary)
plt.title(f"Predicted: {prediction}")
plt.show()'''


while os.path.isfile(f"/Users/edelta076/Desktop/char_recognition/img{img_num}.png") :
    image_path = f"/Users/edelta076/Desktop/char_recognition/img{img_num}.png"
    img = preprocess_image(image_path)
    pred = model.predict(img)
    #print(f"Prediction for image {img_num} is: {np.argmax(pred)}")
    img_to_show = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img_to_show,cmap='gray')
    plt.title(f"Predicted: {np.argmax(pred)}")
    plt.show()
    img_num += 1


'''num_row = 2
num_col = 5
num = 10

fig,axis = plt.subplots(num_row, num_col, figsize = (1.5*num_col,2*num_row))
for i in range(num) :
    ax = axis[i//num_col, i%num_col]
    ax.imshow(x_train[i], cmap='gray')
    ax.set_title('Label: {}'.format(y_train[i]))
plt.tight_layout()
plt.show()'''











