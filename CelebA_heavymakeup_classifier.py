#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf  
 
import cv2
import pickle
from PIL import Image
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,confusion_matrix

from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def preprocess_image(image_path, img_size = 128, img_resize = 64, x = 25, y = 45):
    """
    Crops, resizes and normalizes the target image.
    """

    img = cv2.imread(image_path)
    img = img[y:y+img_size, x:x+img_size]
    img = cv2.resize(img, (img_resize, img_resize))
    img = np.array(img, dtype='float32')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img /= 255.0 # Normalization to [0.,1.]

    return img

def get_images(imgs_ids):
    """
    Returns the list of images corresponding to the given IDs.
    """
    imgs = []

    for i in imgs_ids:
        image_path ='./data/celeba/imgs/' + i
        imgs.append(preprocess_image(image_path))

    return imgs


def get_data():
    """
    Returns data.
    """
    file_path = "./data/celeba/list_attr_celeba.csv"
    df = pd.read_csv(file_path, header = 0, index_col = 0).replace(-1,0)

    labels = df['Heavy_Makeup'].values  # label is attractive
    biases = df['Male'].values        # bias is gender
    
    imgs = get_images(df.index)
    imgs = np.array(imgs)
    
    imgs_train = imgs[:int(len(imgs)*0.80)]
    labels_train = labels[:int(len(labels)*0.80)]
    biases_train = biases[:int(len(biases)*0.80)]
    
    imgs_test = imgs[int(len(imgs)*0.80):]
    labels_test = labels[int(len(labels)*0.80):]
    biases_test = biases[int(len(biases)*0.80):]
    
    imgs_test_male = imgs_test[biases_test==1]
    labels_test_male  = labels_test[biases_test==1]
    biases_test_male  = biases_test[biases_test==1]
    
    imgs_test_female = imgs_test[biases_test==0]
    labels_test_female  = labels_test[biases_test==0]
    biases_test_female  = biases_test[biases_test==0]
    
    data = {
            'imgs_train': imgs_train, 
            'labels_train': labels_train, 

            'imgs_test': imgs_test, 
            'labels_test': labels_test, 

            'imgs_test_male': imgs_test_male, 
            'labels_test_male': labels_test_male, 

            'imgs_test_female': imgs_test_female, 
            'labels_test_female': labels_test_female
           }

    
    return data

data = get_data()


def read_data(file_name):
    """
    Reads file_name.pickle and returns its content.
    """
    with (open((file_name+'.pickle'), "rb")) as openfile:
        while True:
            try:
                objects=pickle.load(openfile)
            except EOFError:
                break
    return objects

def save_data(file_name, data):
    """
    Saves data on file_name.pickle.
    """
    with open((file_name+'.pickle'), 'wb') as openfile:
        pickle.dump(data, openfile)


print(data['imgs_train'].shape, data['labels_train'].shape)
print(data['imgs_test'].shape, data['labels_test'].shape)
print(data['imgs_test_male'].shape, data['labels_test_male'].shape)
print(data['imgs_test_female'].shape, data['labels_test_female'].shape)


file_name = 'heavymakeup_imgs_train'
save_data(file_name, data['imgs_train'])

file_name = 'heavymakeup_labels_train'
save_data(file_name, data['labels_train'])

file_name = 'heavymakeup_imgs_test'
save_data(file_name, data['imgs_test'])

file_name = 'heavymakeup_labels_test'
save_data(file_name, data['labels_test'])

file_name = 'heavymakeup_imgs_test_male'
save_data(file_name, data['imgs_test_male'])

file_name = 'heavymakeup_labels_test_male'
save_data(file_name, data['labels_test_male'])

file_name = 'heavymakeup_imgs_test_female'
save_data(file_name, data['imgs_test_female'])

file_name = 'heavymakeup_labels_test_female'
save_data(file_name, data['labels_test_female'])


# Read data
imgs_train = read_data('heavymakeup_imgs_train')
labels_train = read_data('heavymakeup_labels_train')

imgs_test = read_data('heavymakeup_imgs_test')
labels_test = read_data('heavymakeup_labels_test')

imgs_test_male = read_data('heavymakeup_imgs_test_male')
labels_test_male = read_data('heavymakeup_labels_test_male')

imgs_test_female = read_data('heavymakeup_imgs_test_female')
labels_test_female = read_data('heavymakeup_labels_test_female')


class Generator(Sequence) :
  
    def __init__(self, imgs, labels, batch_size) :
        self.imgs = imgs
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self) :
        return (np.ceil(len(self.imgs ) / float(self.batch_size))).astype(np.int64)
  
  
    def __getitem__(self, idx) :
        batch_x = self.imgs[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

        return batch_x, batch_y


training_generator = Generator(imgs_train, labels_train, batch_size = 32)
validation_generator = Generator(imgs_test, labels_test, batch_size = 32)



# Model
inputs = Input(shape=(64, 64, 3))
baseModel = VGG16(include_top=False, weights='imagenet',input_shape=(64,64,3))
baseModel.trainable = False
x = baseModel(inputs,training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(512,activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256,activation='relu')(x)
output = Dense(2,activation='softmax')(x)
model = Model(inputs=inputs,outputs=output)


model.compile(optimizer=RMSprop(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# save checkpoint
checkpointer = ModelCheckpoint(filepath='weights.heavymakeup.hdf5')


# Fit
r = model.fit(training_generator, epochs=5, validation_data = validation_generator, callbacks=[checkpointer])


#load the best model
model.load_weights('weights.heavymakeup.hdf5')


predictions = model.predict(imgs_test)
predictions = np.argmax(predictions,axis=1)
# report test accuracy
test_accuracy = 100 * np.sum(predictions==labels_test) / len(predictions)
print('Model Evaluation')
print('Test accuracy: %.4f%%' % test_accuracy)
print('f1_score:', f1_score(labels_test, predictions))
print('confusion_matrix:', confusion_matrix(labels_test, predictions))


predictions = model.predict(imgs_test_male)
predictions = np.argmax(predictions,axis=1)
# report test accuracy
test_accuracy = 100 * np.sum(predictions==labels_test_male) / len(predictions)
print('Model Evaluation')
print('Test accuracy: %.4f%%' % test_accuracy)
print('f1_score:', f1_score(labels_test_male, predictions))
print('confusion_matrix:', confusion_matrix(labels_test_male, predictions))


predictions = model.predict(imgs_test_female)
predictions = np.argmax(predictions,axis=1)
# report test accuracy
test_accuracy = 100 * np.sum(predictions==labels_test_female) / len(predictions)
print('Model Evaluation')
print('Test accuracy: %.4f%%' % test_accuracy)
print('f1_score:', f1_score(labels_test_female, predictions))
print('confusion_matrix:', confusion_matrix(labels_test_female, predictions))