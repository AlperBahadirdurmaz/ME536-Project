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


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
path='C:\\Users\\alper\\Desktop\\data\\Images\\train'
myList = os.listdir(path)
os.chdir('C:\\Users\\alper\\Desktop\\data\\Images\\train')
myList.pop(0)
if os.path.isdir('train\\0\\') is False:
    os.mkdir('train')
    os.mkdir('valid')
    os.mkdir('test')
    for i in range(0,len(myList)):
        shutil.move(f'{i}','train')
        os.mkdir(f'valid\\{i}')
        os.mkdir(f'test\\{i}')
        valid_samples=  random.sample(os.listdir(f'train\\{i}'),8)
        for j in valid_samples:
            shutil.move(f'train\\{i}\\{j}',f'valid\\{i}')
        test_samples= random.sample(os.listdir(f'train\\{i}'),3)
        for k in test_samples:
            shutil.move(f'train\\{i}\\{k}',f'test\\{i}')
os.chdir('..\\..')
train_path='C:\\Users\\alper\\Desktop\\data\\Images\\train\\train'
valid_path='C:\\Users\\alper\\Desktop\\data\\Images\\train\\valid'
test_path='C:\\Users\\alper\\Desktop\\data\\Images\\train\\test'

train_batches= ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(directory=train_path, target_size=(224,224),classes=['Apple','Banana','Orange'],batch_size=10)
valid_batches= ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(directory=valid_path, target_size=(224,224),classes=['Apple','Banana','Orange'],batch_size=8)    
test_batches= ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(directory=test_path, target_size=(224,224),classes=['Apple','Banana','Orange'],batch_size=3)






model_12 = load_model('model.h5')

model1=Sequential()
for layer in model_12.layers[:-1]:
    model1.add(layer)
    
model1.add(Dense(units=x1 ,activation='sigmoid'))
        
model1.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
model1.fit(x=train_batches,validation_data=valid_batches, epochs=10, verbose=2)

# Predict using fine-tuned VGG16 model
predictions= model.predict(x=test_batches, verbose=1)














