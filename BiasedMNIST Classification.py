#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.layers import Input,Conv2D,Activation,MaxPool2D,Flatten,Add,Dense,Lambda,concatenate

from tensorflow.keras.losses import binary_crossentropy, mean_squared_error

from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

test_acc_clean = tf.metrics.SparseCategoricalAccuracy()
test_acc_fgsm = tf.metrics.SparseCategoricalAccuracy()

from sklearn import metrics


red = np.array([255,0,0], dtype='float32')
green = np.array([0,255,0], dtype='float32')


def add_color(images, labels, e=0.5):
        
    new_images = []
    new_labels = []
    new_colors = []
    
    images = np.tile(images[:, :, :, np.newaxis], 3)

    
    for image, label in zip(images, labels):
        
       # Assign a binary label y to the image based on the digit
        binary_label = 0 if label < 5 else 1
        
        # Flip label with 25% probability
        if np.random.uniform() < 0.25:
            binary_label = binary_label ^ 1
            
        # Color the image either red or green according to its possibly flipped label
        color_red = binary_label == 0
        
        # Flip the color with a probability e that depends on the environment
        if np.random.uniform() < e:
            color_red = not color_red
            
        # Color image based on color_red label
        if color_red:
            image[:, :, 0] = image[:, :, 0] / 255 * red[0]
            image[:, :, 1] = image[:, :, 1] / 255 * red[1]
            image[:, :, 2] = image[:, :, 2] / 255 * red[2]
        else:
            image[:, :, 0] = image[:, :, 0] / 255 * green[0]
            image[:, :, 1] = image[:, :, 1] / 255 * green[1]
            image[:, :, 2] = image[:, :, 2] / 255 * green[2]

        new_images.append(image)
        new_labels.append(binary_label)
        new_colors.append(color_red)

    return np.asarray(new_images), np.asarray(new_labels), np.asarray(new_colors)


# Prepare datasets.
(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_tr, y_tr, c_tr = add_color(x_train, y_train, e=0.2)
x_te, y_te, c_te = add_color(x_test, y_test, e=0.9)
x_tr, x_te = x_tr.astype('float32')/255., x_te.astype('float32')/255.

batch_size = 64  # set batch size

# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the test dataset.
test_dataset = tf.data.Dataset.from_tensor_slices((x_te, y_te))
test_dataset = test_dataset.batch(batch_size)

test_dataset_red = tf.data.Dataset.from_tensor_slices((x_te[c_te==1], y_te[c_te==1]))
test_dataset_red = test_dataset_red.batch(batch_size)

test_dataset_green = tf.data.Dataset.from_tensor_slices((x_te[c_te==0], y_te[c_te==0]))
test_dataset_green = test_dataset_green.batch(batch_size)


# Classification model

input=Input(shape=(28,28,3))

conv1=Conv2D(32,3,name='conv1')(input)
act1=Activation('relu',name='act1')(conv1)

conv2=Conv2D(32,3,name='conv2',padding='same')(act1)
act2=Activation('relu',name='act2')(conv2)

conv3=Conv2D(32,3,name='conv3',padding='same')(act2)
act3=Activation('relu',name='act3')(conv3)

pool2=MaxPool2D(4,name='pool2')(act3)
flat2=Flatten(name='flat2')(pool2)

output=Dense(2,name='digit')(flat2)
model=tf.keras.models.Model(input,output)

loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam(learning_rate=0.001)
model.compile(loss=loss_object, metrics=['accuracy'], optimizer=optimizer)


# Tratining
history=model.fit(train_dataset, epochs=2)

# Test
history=model.evaluate(test_dataset)