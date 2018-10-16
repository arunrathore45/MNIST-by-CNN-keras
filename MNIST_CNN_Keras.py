import numpy as np 
from PIL import Image
import numpy as np
from keras import layers
from keras.layers import Input,Dense, MaxPooling2D,Dropout,Conv2D,BatchNormalization,Flatten,ZeroPadding2D,Activation
from keras.models import Model
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import cv2
from keras.utils import np_utils
import pandas as pd 

def data_preprocessing():
	train=pd.read_csv("C:/Users/Arun Rathore/Desktop/mnist/mnist_train.csv")
	x_final=np.array(train.iloc[0:,1:]).reshape(train.shape[0],28,28,1)
	k=int(0.8*x_final.shape[0])
	x_train=x_final[0:8000]
	y=np.array(train.iloc[:,0])
	temp=[]
	for i in range(10000):
		b=np.zeros((10))
		b[y[i]]=1
		temp.append(b)
	y_train=np.array(temp[0:8000])	
	x_test=x_final[3000:4000]
	y_test=np.array(temp[3000:4000])
	return x_train,y_train,x_test,y_test
x_train,y_train,x_test,y_test=data_preprocessing()

def model(input_shape):
	X_inputs=Input(input_shape)
	X=ZeroPadding2D((3,3))(X_inputs)
	X=Conv2D(32,(7,7),strides=(1,1),kernel_initializer="he_normal")(X)
	X=BatchNormalization(axis=-1)(X)
	X=Activation('relu')(X)
	X=MaxPooling2D((2,2))(X)
	X=Flatten()(X)
	X=Dropout(0.4)(X)
	X = Dense(10, activation='sigmoid')(X)
	model = Model(inputs = X_inputs, outputs = X)
	return model

mnist=model((x_train.shape[1],x_train.shape[1],1))
mnist.compile(optimizer="adam",loss="mean_squared_error",metrics=["accuracy"])
mnist.fit(x=x_train,y=y_train,epochs=15,batch_size=32)
preds = mnist.evaluate(x=x_test,y=y_test)
print("Loss",str(preds[0]))
print("Accuracy",str(preds[1]))