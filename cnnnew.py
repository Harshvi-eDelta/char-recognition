import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
import cv2
from tensorflow.keras.utils import to_categorical

model = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test) = model.load_data()
print(y_train.shape)

#print(x_train[0])
x_train = tf.keras.utils.normalize(x_train,axis = 1)
x_test = tf.keras.utils.normalize(x_test,axis = 1)
#print(x_train[0])

#print(y_train[5])

x_trainr = np.array(x_train).reshape(-1,28,28,1)
x_testr = np.array(x_test).reshape(-1,28,28,1)
#print(x_test.shape)

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)
print(y_train_one_hot[0])

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape = x_testr.shape[1:]),
    tf.keras.layers.MaxPooling2D((2, 2)),  
     
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

#print(model.summary())
#print(len(x_trainr))

model.compile(loss = tf.keras.losses.sparse_categorical_crossentropy,optimizer = "adam",metrics = ['accuracy'])
model.fit(x_trainr,y_train, epochs = 5, validation_split=0.3)

test_loss,test_acc = model.evaluate(x_testr,y_test)
print(test_loss)
print(test_acc)

pred = model.predict(x_testr)
#print(pred)
#print(np.argmax(pred[0]))
#plt.imshow(x_test[0])
#plt.show()

#print(np.argmax(pred[128]))
#plt.imshow(x_test[128])
#plt.show()

img = cv2.imread('Untitled.png')
plt.imshow(img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_,thre_img = cv2.threshold(gray,100,225,cv2.THRESH_BINARY)
#img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
img = cv2.resize(thre_img,(28,28),interpolation = cv2.INTER_AREA)
img = tf.keras.utils.normalize(img,axis = 1)
img = np.array(img).reshape(-1,28,28,1)
pred_img = model.predict(img)
print(np.argmax(pred_img))
plt.show()


 
