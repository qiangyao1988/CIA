#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf  
 
import cv2
import pickle
import time
import math
import random
import os
import json
import sys

from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.metrics import f1_score,confusion_matrix
from collections import OrderedDict

from tensorflow.keras.utils import Sequence
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Flatten, Dropout
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


# Read data 
train_data = read_data("train_data")
train_data['attributes'][20], train_data['attributes'][18], train_data['attributes'][2]


def get_image(image_path, model_name, img_size = 128, img_resize = 64, x = 25, y = 45):
    """
    Crops, resizes and normalizes the target image.
        - If model_name == Dense, the image is returned as a flattened numpy array with dim (64*64*3)
        - otherwise, the image is returned as a numpy array with dim (64,64,3)
    """

    img = cv2.imread(image_path)
    img = img[y:y+img_size, x:x+img_size]
    img = cv2.resize(img, (img_resize, img_resize))
    img = np.array(img, dtype='float32')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img /= 255.0 # Normalization to [0.,1.]

    if model_name == "Dense" :
        img = img.ravel()
    
    return img


def create_image_batch(labels, model_name):
    """
    Returns the list of images corresponding to the given labels.
    """
    imgs = []
    imgs_id = [item[0] for item in labels]

    for i in imgs_id:
        image_path ='./data/celeba/imgs/' + i
        imgs.append(get_image(image_path, model_name))

    return imgs


def batch_generator(batch_dim, test_labels, model_name):
    """
    Batch generator using the given list of labels.
    """
    while True:
        batch_imgs = []
        labels = []
        for label in (test_labels):
            labels.append(label)
            if len(labels) == batch_dim:
                batch_imgs = create_image_batch(labels, model_name)
                batch_labels = [x[1] for x in labels]
                yield np.asarray(batch_imgs), np.asarray(batch_labels)
                batch_imgs = []
                labels = []
                batch_labels = []
        if batch_imgs:
            yield np.asarray(batch_imgs), np.asarray(batch_labels)


def generate_interpolation_image(train_data, batch_size = 32, num=32, target_attribute = 20, ratio = 3):
    new_images = []
    new_labels = []
    
    batch_gen = batch_generator(train_data['batch_size'], train_data['test_img_ids'], model_name = 'Conv')
    ratios = np.linspace(0, ratio, num=num)
    
    for k in range(2000):
        images, labels = next(batch_gen)  
        
        for i in range(batch_size):
            for j in range(num):
                new_images.append(images[i])
                original_label = labels[i].copy()
                original_label[target_attribute] = ratios[j]
                new_labels.append(original_label)

    return np.asarray(new_images), np.asarray(new_labels)

new_images, new_labels = generate_interpolation_image(train_data, batch_size = 32, num=2, target_attribute = 20, ratio = 3)


# Model Hyper-parameters
label_dim = 40
image_dim = [64, 64, 3]
latent_dim = 128
beta = 0.65

#########################
#        ENCODER        #
#########################

class Encoder(tf.keras.Model):

    def __init__(self, latent_dim):

        super(Encoder, self).__init__()

        self.enc_block_1 = Conv2D( 
                            filters=32, 
                            kernel_size=3, 
                            strides=(2, 2), 
                            padding = 'same',
                            kernel_initializer=he_normal())


        self.enc_block_2 = Conv2D( 
                      filters=64, 
                      kernel_size=3, 
                      strides=(2, 2), 
                      padding = 'same',
                      kernel_initializer=he_normal())



        self.enc_block_3 = Conv2D( 
                      filters=128, 
                      kernel_size=3, 
                      strides=(2, 2), 
                      padding = 'same',
                      kernel_initializer=he_normal())



        self.enc_block_4 = Conv2D( 
                      filters=256, 
                      kernel_size=3, 
                      strides=(2, 2), 
                      padding = 'same',
                      kernel_initializer=he_normal())

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(latent_dim + latent_dim)  


    def __call__(self, conditional_input, latent_dim, is_train):
        # Encoder block 1
        x = self.enc_block_1(conditional_input)
        x = BatchNormalization(trainable = is_train)(x)
        x = tf.nn.leaky_relu(x)
        # Encoder block 2
        x = self.enc_block_2(x)
        x = BatchNormalization(trainable = is_train)(x)
        x = tf.nn.leaky_relu(x)
        # Encoder block 3
        x = self.enc_block_3(x)
        x = BatchNormalization(trainable = is_train)(x)
        x = tf.nn.leaky_relu(x)
        # Encoder block 4
        x = self.enc_block_4(x)
        x = BatchNormalization(trainable = is_train)(x)
        x = tf.nn.leaky_relu(x)   

        x = self.dense(self.flatten(x))

        return x


#########################
#        DECODER        #
#########################

class Decoder(tf.keras.Model):
    

    def __init__(self, batch_size = 32):

        super(Decoder, self).__init__()

        self.batch_size = batch_size
        self.dense = tf.keras.layers.Dense(4*4*self.batch_size*8)
        self.reshape = tf.keras.layers.Reshape(target_shape=(4, 4, self.batch_size*8))

        self.dec_block_1 = Conv2DTranspose(
                filters=256,
                kernel_size=3,
                strides=(2, 2),
                padding='same',
                kernel_initializer=he_normal())

        self.dec_block_2 = Conv2DTranspose(
                filters=128,
                kernel_size=3,
                strides=(2, 2),
                padding='same',
                kernel_initializer=he_normal())

        self.dec_block_3 = Conv2DTranspose(
                filters=64,
                kernel_size=3,
                strides=(2, 2),
                padding='same',
                kernel_initializer=he_normal())

        self.dec_block_4 = Conv2DTranspose(
                filters=32,
                kernel_size=3,
                strides=(2, 2),
                padding='same',
                kernel_initializer=he_normal())

        self.dec_block_5 = Conv2DTranspose(
                filters=3, 
                kernel_size=3, 
                strides=(1, 1), 
                padding='same',
                kernel_initializer=he_normal())

    def __call__(self, z_cond, is_train):
        # Reshape input
        x = self.dense(z_cond)
        x = tf.nn.leaky_relu(x)
        x = self.reshape(x)
        # Decoder block 1
        x = self.dec_block_1(x)
        x = BatchNormalization(trainable = is_train)(x)
        x = tf.nn.leaky_relu(x)
        # Decoder block 2
        x = self.dec_block_2(x)
        x = BatchNormalization(trainable = is_train)(x)
        x = tf.nn.leaky_relu(x)
        # Decoder block 3
        x = self.dec_block_3(x)
        x = BatchNormalization(trainable = is_train)(x)
        x = tf.nn.leaky_relu(x)
        # Decoder block 4
        x = self.dec_block_4(x)
        x = BatchNormalization(trainable = is_train)(x)
        x = tf.nn.leaky_relu(x)

        return self.dec_block_5(x)


#########################
#       Conv-CVAE       #
#########################

class ConvCVAE (tf.keras.Model) :

    def __init__(self, 
        encoder,
        decoder,
        label_dim,
        latent_dim,
        batch_size = 32,
        beta = 1,
        image_dim = [64, 64, 3]):

        super(ConvCVAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.label_dim = label_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.beta = beta
        self.image_dim = image_dim = [64, 64, 3]              


    def __call__(self, inputs, is_train):
    
        input_img, input_label, conditional_input = self.conditional_input(inputs)

        z_mean, z_log_var = tf.split(self.encoder(conditional_input, self.latent_dim, is_train), num_or_size_splits=2, axis=1)    
        z_cond = self.reparametrization(z_mean, z_log_var, input_label)
        logits = self.decoder(z_cond, is_train)

        recon_img = tf.nn.sigmoid(logits)

        # Loss computation #
        latent_loss = - 0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1) # KL divergence
        reconstr_loss = np.prod((64,64)) * tf.keras.losses.binary_crossentropy(tf.keras.backend.flatten(input_img), tf.keras.backend.flatten(recon_img)) # over weighted MSE
        loss = reconstr_loss + self.beta * latent_loss # weighted ELBO loss
        loss = tf.reduce_mean(loss) 

        return {
                    'recon_img': recon_img,
                    'latent_loss': latent_loss,
                    'reconstr_loss': reconstr_loss,
                    'loss': loss,
                    'z_mean': z_mean,
                    'z_log_var': z_log_var
                }


    def conditional_input(self, inputs):
        """ Builds the conditional input and returns the original input images, their labels and the conditional input."""

        input_img = tf.keras.layers.InputLayer(input_shape=self.image_dim, dtype = 'float32')(inputs[0])
        input_label = tf.keras.layers.InputLayer(input_shape=(self.label_dim,), dtype = 'float32')(inputs[1])
        labels = tf.reshape(inputs[1], [-1, 1, 1, self.label_dim]) #batch_size, 1, 1, label_size
        ones = tf.ones([inputs[0].shape[0]] + self.image_dim[0:-1] + [self.label_dim]) #batch_size, 64, 64, label_size
        labels = ones * labels #batch_size, 64, 64, label_size
        conditional_input = tf.keras.layers.InputLayer(input_shape=(self.image_dim[0], self.image_dim[1], self.image_dim[2] + self.label_dim), dtype = 'float32')(tf.concat([inputs[0], labels], axis=3))

        return input_img, input_label, conditional_input


    def reparametrization(self, z_mean, z_log_var, input_label):
        """ Performs the riparametrization trick"""

        eps = tf.random.normal(shape = (input_label.shape[0], self.latent_dim), mean = 0.0, stddev = 1.0)       
        z = z_mean + tf.math.exp(z_log_var * .5) * eps
        z_cond = tf.concat([z, input_label], axis=1) # (batch_size, label_dim + latent_dim)

        return z_cond

# Generative Model
encoder = Encoder(latent_dim)
decoder = Decoder()
CVAE = ConvCVAE(
                encoder,
                decoder,
                label_dim = label_dim,
                latent_dim = latent_dim,
                beta = beta,
                image_dim = image_dim)


# Checkpoint path
checkpoint_root = "./CVAE{}_{}_checkpoint".format(latent_dim, beta)
checkpoint_name = "model"
save_prefix = os.path.join(checkpoint_root, checkpoint_name)

# Define the checkpoint
checkpoint = tf.train.Checkpoint(module=CVAE)


# Restore the latest checkpoint
latest = tf.train.latest_checkpoint(checkpoint_root)
if latest is not None:
    checkpoint.restore(latest)
    print("Checkpoint restored:", latest)
else:
    print("No checkpoint!")


def interpolation(new_images, new_labels, model, batch_size):
    
    generated_images = []
    generated_labels = []
    
    for i in range(0,len(new_images),batch_size):
    
        img = new_images[i:i+batch_size]
        label = new_labels[i:i+batch_size]
        
        model_output = model((img, label), is_train = False)
        img_z = model_output['z_mean']
        z_cond = tf.concat([img_z,label], axis=1)
        logits = model.decoder(z_cond, is_train = False)
        generated = tf.nn.sigmoid(logits)
   
        generated_images.extend(generated)
        generated_labels.extend(label)
        
    generated_images = np.asarray(generated_images) 
    generated_labels = np.asarray(generated_labels) 
        
    return generated_images,generated_labels


generated_images, generated_labels = interpolation(new_images, new_labels, CVAE, batch_size=8)


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

    data = {
        'imgs_train': imgs_train, 
        'labels_train': labels_train,
    }

    return data


data = get_data()


original_imgs = data['imgs_train']
original_labels = data['labels_train']


aug_images = np.concatenate((original_imgs, generated_images), axis=0)
aug_labels = np.concatenate((original_labels, new_labels[:,18]), axis=0)


# Read  data
imgs_test = read_data('heavymakeup_imgs_test')
labels_test = read_data('heavymakeup_labels_test')

imgs_test_male = read_data('heavymakeup_imgs_test_male')
labels_test_male = read_data('heavymakeup_labels_test_male')

imgs_test_female = read_data('heavymakeup_imgs_test_female')
labels_test_female = read_data('heavymakeup_labels_test_female')


print('test set:', imgs_test.shape, labels_test.shape)
print('male set:', imgs_test_male.shape, labels_test_male.shape)
print('femal set:', imgs_test_female.shape, labels_test_female.shape)


# Classification Model
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


training_generator = Generator(aug_images, aug_labels, batch_size=32)


model.compile(optimizer=RMSprop(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# save checkpoint
checkpointer = ModelCheckpoint(filepath='weights.heavymakeup_fair.hdf5')

r = model.fit(training_generator, epochs=2, callbacks=[checkpointer])

#load the best model
model.load_weights('weights.heavymakeup_fair.hdf5')

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