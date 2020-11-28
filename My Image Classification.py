#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# All The Import Module
import pandas as pd
import numpy as np
import PIL
import os
from PIL import Image
from skimage.io import imread, imshow
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


# My Training Image Path: C:\\Users\\win 10\\Downloads\\Image Classification\\Images
# My Testing Image Path: C:\\Users\\win 10\\Downloads\\Image Classification\\Testimage

# I have Different size of Images, so First Operation is Do all the Images size is constant.
# Set My Path
TrainImagePath = r'C:\\Users/win 10/Downloads/Image Classification/Images'
TestImagePath = r'C:\\Users/win 10/Downloads/Image Classification/Testimage'

# List Of all File In My Dir.
TrainI = os.listdir(TrainImagePath)
TestI = os.listdir(TestImagePath)


# In[ ]:


# Note: ML model At the time of Training Given data is same for prediction time (sequence, feature size, type, etc)
# so now we convert our both data Training and Testing Same Size of Image.


#Reshape Traning Image
for file in TrainI:
    Fimg = TrainImagePath + '/' + file
    img = Image.open(Fimg)
    img = img.resize((800,800))
    img.save(Fimg)

#Reshape Testing Image.
for file in TestI:
    Fimg = TestImagePath + '/' + file
    img = Image.open(Fimg)
    img = img.resize((800,800))
    img.save(Fimg)


# In[ ]:


# on the Basis of my training images, i have CSV file that contains name of Image(name of person) and image's file name, so  first read it.
TrainC = pd.read_csv(r'Image Classification\Mytrain.csv')
TrainC.head()
Ytrain = TrainC['name']
TestC = pd.read_csv(r'Image Classification\Mytest.csv')
TestC.head()
Ytest = TestC['name']


# In[ ]:


# Reading Image and Extreat in Metrics form (Grey Color).
# first I am trying, how it's work.

img = imread(TestImagePath + '/' + '1.jpg', as_gray = True)
imshow(img)
# hey it works
print(img.shape)
print(img[0][0])
print(len(img))
X1 = np.reshape(img, img.shape[0]*img.shape[1])


# In[ ]:


print(X1.shape)
totalPixel = 800*800
print(totalPixel)


# In[ ]:


'''
1.Now, we have one image matrix, but we required each image matrix. Simple we use Loop
2.ML algoritham we do not provide training in matrix form again we use numpy and transform into single dimension array.(Test and Training data both.)
  because of we provide as a Feature Of image(Each of columns)
3.when Training is completed we provide test data.
4.compareing result, confusion matrix, evaluation, apply diffrent algoritham and fine tuneing them.
'''
#Create Training data from images in form of data frame
#because all ML Algorithm fit function takes training data as
#data Frame only

rows=[]
for i in range(len(TrainI)):
    Fimg = TrainImagePath + '/' + TrainI[i]
    image_black=imread(Fimg,as_gray=True)
    X1=np.reshape(image_black, totalPixel)
    rows.append(X1)
    print(i) # shows what image actually process.
Xtrain=pd.DataFrame(rows)
# so we have Both Training Data (Xtrain and Ytrain).
# this Process we apply for testing data


# In[ ]:


# now we prepare data for test

Trows=[]
for i in range(len(TestI)):
    Fimg = TestImagePath + '/' + TestI[i]
    image_black=imread(Fimg,as_gray=True)
    X1=np.reshape(image_black, totalPixel)
    Trows.append(X1)
    print(i) # shows what image actually process.
Xtest=pd.DataFrame(Trows)


# In[ ]:


# finally we have all data for Xtest, Ytest for Testing purpose
# we have all data for Xtrain, Ytrain for ML model training
# machine Learning is an experimental so if we not satisfying result then we apply different algoritham
# but now we apply KNeighborsClassifier

knn=KNeighborsClassifier()
knn.fit(Xtrain,Ytrain)
Ypredict = knn.predict(Xtest)
print(knn.score(Xtest,Ytest))

#result = pd.DataFrame([Ypredict, Ytest], columns = ['Predicted','Actual'])


# In[ ]:


result = pd.DataFrame({
    'Predicted':Ypredict,
    'Actual':Ytest
})
result.to_csv()


# In[ ]:




