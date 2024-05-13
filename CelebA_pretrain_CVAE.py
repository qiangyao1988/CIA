#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf  
 
# other imports
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt

import time
import math
import random
import os
import pickle
import json
import sys

from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, Conv2DTranspose
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.initializers import he_normal

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def save_data(file_name, data):
    """
    Saves data on file_name.pickle.
    """
    with open((file_name+'.pickle'), 'wb') as openfile:
        pickle.dump(data, openfile)


class CelebADataset(Sequence):


    def __init__(self, train_size, batch_size, mode = 'train'):

        self.train_img_ids, self.test_img_ids, self.attributes = self.load(train_size)
        self.batch_size = batch_size
        self.mode = mode
        self.train_size = len(self.train_img_ids)
        self.save_test_set()

    def load(self, train_dim):
        """ 
        Loads all image IDs and the attributes and splits the dataset into training set and test set.
            
        Returns:
            - train_img_ids [list]
            - test_img_ids [list]
            - attributes [list]
            
        """

        print("Loading images id and attributes...")

        file_path = "./data/celeba/list_attr_celeba.csv"
        df = pd.read_csv(file_path, header = 0, index_col = 0).replace(-1,0)
        attributes = [x for x in df.columns] 
        od = OrderedDict(df.to_dict('index'))
        img_ids = OrderedDict()
        for k,v in od.items():
            img_id = [np.float32(x) for x in v.values()]
            img_ids[k] = img_id
        print("img_ids: {} \nAttributes: {} \n".format(len(img_ids), len(attributes)))

        #Splitting
        print("Splitting dataset...\n")
        n_train = int(len(img_ids) * train_dim)
        list_img_ids = list(img_ids.items())
        train_img_ids = list_img_ids[:n_train]
        test_img_ids = list_img_ids[n_train:]

        print("Train set dimension: {} \nTest set dimension: {} \n".format(len(train_img_ids), len(test_img_ids)))

        return train_img_ids, test_img_ids, attributes


    def next_batch(self, idx):
        """
        Returns a batch of images with their ID as numpy arrays.
        """    
        
        batch_img_ids = [x[1] for x in self.train_img_ids[idx * self.batch_size : (idx + 1) * self.batch_size]]
        images_id = [x[0] for x in self.train_img_ids[idx * self.batch_size : (idx + 1) * self.batch_size]]
        batch_imgs = self.get_images(images_id) 
        
        return np.asarray(batch_imgs, dtype='float32'), np.asarray(batch_img_ids, dtype='float32')


    def preprocess_image(self, image_path, img_size = 128, img_resize = 64, x = 25, y = 45):
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


    def get_images(self,imgs_id):
        """
        Returns the list of images corresponding to the given IDs.
        """
        imgs = []

        for i in imgs_id:
            image_path ='./data/celeba/imgs/' + i
            imgs.append(self.preprocess_image(image_path))

        return imgs
    
    def save_test_set(self):
        """
        Saves a json file with useful information for teh test phase:
            - training size
            - test images IDs
            - attributes
            - batch size
        """

        try:
            test_data = {
                'train_size' : self.train_size,
                'test_img_ids' : self.test_img_ids,
                'attributes' : self.attributes,
                'batch_size' : self.batch_size
            }

            file_path = "test_data"
            save_data(file_path, test_data)
        except:
            raise
        print("Test img_ids successfully saved.")
        

    def shuffle(self):
        """
        Shuffles the training IDs.
        """
        self.train_img_ids = random.sample(self.train_img_ids, k=self.train_size)
        print("IDs shuffled.")


    def __len__(self):
        return int(math.ceil(self.train_size / float(self.batch_size)))


    def __getitem__(self, index):
        return self.next_batch(index)


# Training configuration
learning_rate = 0.001
train_size = 0.95
batch_size = 32

# Hyper-parameters
label_dim = 40
image_dim = [64, 64, 3]
latent_dim = 128
beta = 0.65

dataset = CelebADataset(train_size = train_size, batch_size = batch_size)


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

# Model
encoder = Encoder(latent_dim)
decoder = Decoder()
model = ConvCVAE(
                encoder,
                decoder,
                label_dim = label_dim,
                latent_dim = latent_dim,
                beta = beta,
                image_dim = image_dim)

# Optiizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


# Checkpoint path
checkpoint_root = "./CVAE{}_{}_checkpoint".format(latent_dim, beta)
checkpoint_name = "model"
save_prefix = os.path.join(checkpoint_root, checkpoint_name)

# Define the checkpoint
checkpoint = tf.train.Checkpoint(module=model)

# Restore the latest checkpoint
latest = tf.train.latest_checkpoint(checkpoint_root)

if latest is not None:
    checkpoint.restore(latest)
    print("Checkpoint restored:", latest)
else:
    print("No checkpoint!")


#######################
# Train Step Function #
#######################

def train_step(data, model, optimizer):


    with tf.GradientTape() as tape:
        
        model_output = model(data, is_train = True)

    trainable_variables = model.trainable_variables
    grads = tape.gradient(model_output['loss'], trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))

    total_loss = model_output['loss'].numpy().mean()
    recon_loss = model_output['reconstr_loss'].numpy().mean()
    latent_loss = model_output['latent_loss'].numpy().mean()

    return total_loss, recon_loss, latent_loss


train_losses = []
train_recon_errors = []
train_latent_losses = []
loss = []
reconstruct_loss = []
latent_loss = []

step_index = 0
n_batches = int(dataset.train_size / batch_size)
n_epochs = 30

print("Number of epochs: {},  number of batches: {}".format(n_epochs, n_batches))


# Epochs Loop
for epoch in range(n_epochs):
    start_time = time.perf_counter()
    dataset.shuffle() # Shuffling

    # Train Step Loop
    for step_index, inputs in enumerate(dataset):
        total_loss, recon_loss, lat_loss = train_step(inputs, model, optimizer)
        train_losses.append(total_loss)
        train_recon_errors.append(recon_loss)
        train_latent_losses.append(lat_loss)

        if step_index + 1 == n_batches:
            break

    loss.append(np.mean(train_losses, 0))
    reconstruct_loss.append(np.mean(train_recon_errors, 0))
    latent_loss.append(np.mean(train_latent_losses, 0))

    exec_time = time.perf_counter() - start_time
    print("Execution time: %0.3f \t Epoch %i: loss %0.4f | reconstr loss %0.4f | latent loss %0.4f"
                        % (exec_time, epoch, loss[epoch], reconstruct_loss[epoch], latent_loss[epoch])) 


    # Save progress every 5 epochs
    if (epoch + 1) % 5 == 0:
        checkpoint.save(save_prefix + "_" + str(epoch + 1))
        print("Model saved:", save_prefix)
            
# Save the final model                
checkpoint.save(save_prefix)
print("Model saved:", save_prefix)

