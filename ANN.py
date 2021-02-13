
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
warnings.simplefilter(action='ignore', category=FutureWarning)   
#%matplotlib inline
from elevate import elevate


q=os.getcwd()
print(q)
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
"""
def plotImages(images_arr):
    fig, axes = plt.subplots(1,10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                        height_shift_range=0.1, shear_range=0.15, zoom_range=0.1,
                        channel_shift_range=10., horizontal_flip=True,)
chosen_image = random.choice(os.listdir('C:\\Users\\alper\\Desktop\\data\\Data\\Fruits\\train'))
image_path =  'C:\\Users\\alper\\Desktop\\data\\Data\\Fruits\\train\\'+chosen_image

print('{} \n'.format(chosen_image))
print(image_path)
elevate(show_console=False)
image= np.expand_dims(plt.imread(image_path,format='jpeg'),0)
aug_iter= gen.flow(image)
aug_images=[next(aug_iter)[0].astype(np.unit8)] """






model_16=tf.keras.applications.vgg16.VGG16()

model=Sequential()
for layer in model_16.layers[:-1]:
    model.add(layer)
    
model.add(Dense(units=3,activation='sigmoid'))
        
model.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x=train_batches,validation_data=valid_batches, epochs=10, verbose=2)

# Predict using fine-tuned VGG16 model
predictions= model.predict(x=test_batches, verbose=1)

import os.path
if os.path.isfile('C:\\Users\\alper\\Desktop\\fruit\\ANN') is false:
    model.save('model.h1')





