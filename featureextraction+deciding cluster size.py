# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 23:37:06 2021

@author: alper
"""
# Code of your program here
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.applications import VGG16
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
from os import chdir
import pandas as pd
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)   
#%matplotlib inline
from elevate import elevate
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input 
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

path='C:\\Users\\alper\\Desktop\\data\\Data\\Fruits\\train'
myList = os.listdir(path)
os.chdir('C:\\Users\\alper\\Desktop\\data\\Data\\Fruits\\train')
train_path='C:\\Users\\alper\\Desktop\\data\\Data\\Fruits\\train'
img=[]
model_16=tf.keras.applications.vgg16.VGG16()

model=Sequential()
for layer in model_16.layers[:-3]:
    model.add(layer) 
x_list=[]
y_list=[]
print(myList)
myList.pop()
for i in range(len(myList)):
    list1= os.listdir(f'C:\\Users\\alper\\Desktop\\data\\Data\\Fruits\\train\\{i}')
    list1.pop(0)
    img.append(list1)
    print(len(img[i]))
    for j in range(len(img[i])-1):
        z='C:\\Users\\alper\\Desktop\\data\\Data\\Fruits\\train\\{}\\{}'.format(i,img[i][j])
        if z=='C:\\Users\\alper\\Desktop\\data\\Data\\Fruits\\train\\{}\\desktop.ini'.format(i):
            continue
        else:
            img11=image.load_img(z, target_size=[224,224])
            x=image.img_to_array(img11)
            x=np.expand_dims(x,axis=0)
            x=preprocess_input(x)
            x.reshape(150528,1)
            x_list.append(x)
            y= model.predict(x)
            y_list.append(y)
def SVD536(Mn, DebugMode = False):
  A=np.array([[2,1],[1,5]])
  t=type(A)
  if (not isinstance(Mn,t)):
    return []
  U,S,VT=np.linalg.svd(Mn,full_matrices=False)
  r=np.linalg.matrix_rank(Mn)
  A=((S*S).sum())**(0.5)
  S_1=np.zeros(r)
  for i in range(r):
    S_1[i]=S[i]/A
    if S_1[i]<0.1:
      S[i:]=0
      break
  M=np.matmul(U,np.matmul(np.diag(S),VT))
  return M
final=np.zeros((25088,160))
for i in range(len(y_list)):
    final[:,i]=y_list[i]
reduct= SVD536(final)
r=np.linalg.matrix_rank(reduct) 
#standardize the 
df1=pd.DataFrame(data=reduct)
df1 = StandardScaler().fit_transform(df1)
df2=pd.DataFrame(data=final)
df2 = StandardScaler().fit_transform(df2)


    
#KMeans class




def plot_optimum_cluster(data):
  #set a list to append the iter values
  iter_num = []
  k_cluster_number=[]
  for i in range(1, 15):
      #perform kmeans to get best cluster value using elbow method
      kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42, max_iter = 300)
      kmeans.fit(data)
      iter_num.append(kmeans.inertia_)
  for i in range(1, 13):
          change=(iter_num[i-1]-iter_num[i])-(iter_num[i]-iter_num[i+1])
          k_cluster_number.append(change)
  k_cluster_number1=k_cluster_number.copy()
  a=max(k_cluster_number1)
  a1=k_cluster_number1.index(a)
  k_cluster_number1.pop(a1)
  b=max(k_cluster_number1)
  b1=k_cluster_number1.index(b)
  k_cluster_number1.pop(b1)
  for i in range(12):
          if k_cluster_number[i]==b:
              Ideal_k=i+2
      #plot the optimum graph
  plt.plot(range(1, 15), iter_num)
  plt.title('The Elbow method of determining number of clusters')
  plt.xlabel('Number of clusters')
  plt.ylabel('iter_num')
  plt.show()
  return Ideal_k

'''
The optimum cluster for our dataset according to 
cluster the dataset is 3
'''

x1=plot_optimum_cluster(df1)
x2=plot_optimum_cluster(df2)














