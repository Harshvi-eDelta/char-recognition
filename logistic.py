import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
import cv2

digits = pd.read_csv("./MNIST_CSV/mnist_train.csv")
print(digits.shape)
print(digits.head())

x = digits.drop(columns=['5'])
y = digits['5']
print(x)

x1 = x.values.reshape(-1,28,28,1)
plt.imshow(x1[6],cmap='gray')
print("image is :" , y[6])
#plt.show()

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.3)
model = LogisticRegression()
model.fit(x_train,y_train)
pred = model.predict(x_test)
ac = accuracy_score(y_test,pred)
print("accuracy score :",ac)
cm = confusion_matrix(y_test,pred)
print("confusion matrix : ",cm)

x_test_2 = x_test.values.reshape(-1,28,28,1)
print("image is :" , y_test.iloc[1000])
print("prediction for this image is :",model.predict(x_test)[1000])
plt.imshow(x_test_2[1000],cmap='gray')
plt.show()

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
    img = cv2.resize(img, (28, 28))  
    img = img / 255.0  
    img = img.reshape(1, -1)  
    return img

image_path = f"/Users/edelta076/Desktop/char_recognition/img5.png"  
custom_img = preprocess_image(image_path)  
prediction = model.predict(custom_img)
print(f"Prediction for custom image: {prediction[0]}")

img_to_show = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
plt.imshow(img_to_show, cmap='gray')
plt.title(f"Predicted: {prediction[0]}")
plt.show()

