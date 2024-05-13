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
from sklearn.metrics import f1_score
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
file_path = "./data/celeba/list_attr_celeba.csv"
df = pd.read_csv(file_path, header = 0, index_col = 0).replace(-1,0)

# Read test data
imgs_test = read_data('heavymakeup_imgs_test')
labels_test = read_data('heavymakeup_labels_test')
biases_test = read_data('heavymakeup_biases_test')


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


#load the best model
model.load_weights('weights.heavymakeup.hdf5')


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


def generate_interpolation_image(images, labels, batch_size = 32, num=32, target_attribute = 20, ratio = 3):
     
    imgs = {}
    for i in range(batch_size):
        imgs[i] = []
        for j in range(num):
            imgs[i].append(images[i])
        imgs[i] = np.asarray(imgs[i])

    ratios = np.linspace(0, ratio, num=num)
    
    resized_labels = {}
    for i in range(batch_size):
        resized_labels[i] = []
        for j in range(num):
            original_label = labels[i].copy()
            original_label[target_attribute] = ratios[j]
            resized_labels[i].append(original_label)
        resized_labels[i] = np.asarray(resized_labels[i])   

    return imgs, resized_labels


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

# Generarive Model
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


def interpolation(imgs, resized_labels, model, batch_size):
    generated_images = []
    
    for i in range(batch_size):
    
        img = imgs[i]
        label = resized_labels[i]
        model_output = model((img, label), is_train = False)
        img_z = model_output['z_mean']
        z_cond = tf.concat([img_z,label], axis=1)
        logits = model.decoder(z_cond, is_train = False)
        generated = tf.nn.sigmoid(logits)
        generated_images.append(generated)
        
    generated_images = np.asarray(generated_images) 
    
    return generated_images


# ### CGI

def compute_gradients(images, target_index=1):
    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = model(images)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_index]
    return tape.gradient(probs, images)


def plot_gradients(images, index, target_index=1, num=10):
    path_gradients = compute_gradients(images=tf.convert_to_tensor(images[index]), target_index=target_index)
    
    ratios = np.linspace(0, 3, num=num)
    
    pred = model(images[index])
    pred_proba = tf.nn.softmax(pred, axis=-1)[:, target_index]

    plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(ratios, pred_proba)
    ax1.set_title('Target class predicted probability \n over interplote ratio')
    ax1.set_ylabel('model p(target class)')
    ax1.set_xlabel('interplote ratio')
    ax1.set_ylim([0, 1])

    ax2 = plt.subplot(1, 2, 2)
    # Average across interpolation steps
    average_grads = tf.reduce_mean(path_gradients, axis=[1, 2, 3])
    average_grads_norm = (average_grads-tf.math.reduce_min(average_grads))/(tf.math.reduce_max(average_grads)-tf.reduce_min(average_grads))
    ax2.plot(ratios, average_grads_norm)
    ax2.set_title('Average pixel gradients (normalized) \n over interplote ratio')
    ax2.set_ylabel('Average pixel gradients')
    ax2.set_xlabel('interplote ratio')
    ax2.set_ylim([0, 1]);

plot_gradients(generated_data, index=21, target_index=1, num=10)

def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients

@tf.function
def integrated_gradients(images, index, target_index, num=10):
   
    gradient_batches = tf.TensorArray(tf.float32, size=num)

    gradient_batch = compute_gradients(images=tf.convert_to_tensor(images[index]), target_index=target_index)

    gradient_batches = gradient_batches.scatter(tf.range(0, num), gradient_batch)    
    total_gradients = gradient_batches.stack()

    avg_gradients = integral_approximation(gradients=total_gradients)

    integrated_gradients = (images[index][0] - images[index][num-1]) * avg_gradients

    return integrated_gradients


def plot_img_attributions(images, index, target_index, num=10, overlay_alpha=0.4):

    attributions = integrated_gradients(images, index=index, target_index=target_index, num=num)

    # Sum of the attributions across color channels for visualization.
    # The attribution mask shape is a grayscale image with height and width
    # equal to the original image.
    # attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)

    fig, axs = plt.subplots(nrows=1, ncols=4, squeeze=False, figsize=(6, 6))

    axs[0, 0].set_title('Baseline Image')
    axs[0, 0].imshow(images[index][num-1])
    axs[0, 0].axis('off')

    axs[0, 1].set_title('Original image')
    axs[0, 1].imshow(images[index][0])
    axs[0, 1].axis('off')

    axs[0, 2].set_title('Attribution mask')
    axs[0, 2].imshow(attributions*255., cmap='twilight')
    axs[0, 2].axis('off')

    axs[0, 3].set_title('Overlay')
    axs[0, 3].imshow(attributions*255, cmap='hot')
    axs[0, 3].imshow(images[index][0], alpha=overlay_alpha)
    axs[0, 3].axis('off')

    plt.tight_layout()
    return attributions,fig


attributions,_ = plot_img_attributions(generated_data, index=21, target_index=1, num=10, overlay_alpha=0.5)

def single_run_deletion(image,attribution,model):
    
    start = np.copy(image)
    finish = np.zeros(shape=(64,64,3))
    
    HW = 64*64
    n_steps = (HW + 64 - 1) // 64
    scores = np.empty(n_steps + 1)
    
    salient_order = np.flip(np.argsort(np.asarray(attribution).reshape(-1, HW), axis=1), axis=-1)
    
    pred = model.predict(tf.expand_dims(image, 0))
    # score = tf.nn.softmax(pred)
    top = np.argmax(pred, -1)
    
    for i in range(n_steps+1):
        pred = model.predict(tf.expand_dims(start, 0))
        # score = tf.nn.softmax(pred)
        scores[i] = pred[0, top]
        if i < n_steps:
            coords = salient_order[:, 64 * i:64 * (i + 1)]
            start.reshape(1, 3, HW)[0, :, coords] = finish.reshape(1, 3, HW)[0, :, coords]
            
    return scores

def single_run_insertion(image,attribution,model):
    
    finish = np.copy(image)
    start = np.zeros(shape=(64,64,3))
    
    HW = 64*64
    n_steps = (HW + 64 - 1) // 64
    scores = np.empty(n_steps + 1)
    
    salient_order = np.flip(np.argsort(np.asarray(attribution).reshape(-1, HW), axis=1), axis=-1)
    
    pred = model.predict(tf.expand_dims(image, 0))
    # score = tf.nn.softmax(pred)
    top = np.argmax(pred, -1)
    
    for i in range(n_steps+1):
        pred = model.predict(tf.expand_dims(start, 0))
        # score = tf.nn.softmax(pred)
        scores[i] = pred[0, top]
        if i < n_steps:
            coords = salient_order[:, 64 * i:64* (i + 1)]
            start.reshape(1, 3, HW)[0, :, coords] = finish.reshape(1, 3, HW)[0, :, coords]
            
    return scores


### IG

m_steps= 10
alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps) # Generate m_steps intervals for integral_approximation() below.


def interpolate_images(baseline, image, alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x +  alphas_x * delta
    return images

interpolated_images = interpolate_images(baseline=baseline, image=generated_data[21][0], alphas=alphas)


def compute_gradients(images, target_class_idx):
    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = model(images)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    return tape.gradient(probs, images)


@tf.function
def integrated_gradients(baseline,
                         image,
                         target_class_idx,
                         m_steps=50,
                         batch_size=32):
    # 1. Generate alphas.
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

    # Initialize TensorArray outside loop to collect gradients.    
    gradient_batches = tf.TensorArray(tf.float32, size=m_steps+1)

    # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]

        # 2. Generate interpolated inputs between baseline and input.
        interpolated_path_input_batch = interpolate_images(baseline=baseline,
                                                           image=image,
                                                           alphas=alpha_batch)

        # 3. Compute gradients between model outputs and interpolated inputs.
        gradient_batch = compute_gradients(images=interpolated_path_input_batch,
                                           target_class_idx=target_class_idx)

        # Write batch indices and gradients to extend TensorArray.
        gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)    

    # Stack path gradients together row-wise into single tensor.
    total_gradients = gradient_batches.stack()

    # 4. Integral approximation through averaging gradients.
    avg_gradients = integral_approximation(gradients=total_gradients)

    # 5. Scale integrated gradients with respect to input.
    integrated_gradients = (image - baseline) * avg_gradients

    return integrated_gradients

def plot_img_attributions(baseline,
                          image,
                          target_class_idx,
                          m_steps=50,
                          cmap=None,
                          overlay_alpha=0.4):

    attributions = integrated_gradients(baseline=baseline,
                                       image=image,
                                       target_class_idx=1,
                                       m_steps=10)

    # Sum of the attributions across color channels for visualization.
    # The attribution mask shape is a grayscale image with height and width
    # equal to the original image.
    # attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)

    fig, axs = plt.subplots(nrows=1, ncols=4, squeeze=False, figsize=(6, 6))

    axs[0, 0].set_title('Baseline image')
    axs[0, 0].imshow(baseline)
    axs[0, 0].axis('off')

    axs[0, 1].set_title('Original image')
    axs[0, 1].imshow(image)
    axs[0, 1].axis('off')

    axs[0, 2].set_title('Attribution mask')
    axs[0, 2].imshow(attributions*255, cmap=cmap)
    axs[0, 2].axis('off')

    axs[0, 3].set_title('Overlay')
    axs[0, 3].imshow(attributions*255, cmap=cmap)
    axs[0, 3].imshow(image, alpha=overlay_alpha)
    axs[0, 3].axis('off')

    plt.tight_layout()
    return attributions,fig

attributions, _ = plot_img_attributions(image=generated_data[21][0],
                          baseline=baseline,
                          target_class_idx=1,
                          m_steps=10,
                          cmap=plt.cm.inferno,
                          overlay_alpha=0.5)


### Blur IG

from scipy import ndimage


def gaussian_blur(image, sigma):
    """Returns Gaussian blur filtered 3d (WxHxC) image.
    Args:
    image: 3 dimensional ndarray / input image (W x H x C).
    sigma: Standard deviation for Gaussian blur kernel.
    """
    return ndimage.gaussian_filter(image, sigma=[sigma, sigma, 0], mode="constant")


def interpolate_images(image):
    images = []
    sigmas = [2, 1.8, 1.5, 1.4, 1.2, 1, 0.8, 0.5, 0.2, 0]
    for sig in sigmas:
        images.append(gaussian_blur(image, sig))
    return images


interpolated_images = interpolate_images(image=generated_data[21][0])


def compute_gradients(images, target_class_idx):
    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = model(images)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    return tape.gradient(probs, images)


path_gradients = compute_gradients(
    images=tf.convert_to_tensor(interpolated_images, dtype=tf.float32),
    target_class_idx=1)


def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients


avg_gradients = integral_approximation(
    gradients=path_gradients)

images=tf.convert_to_tensor(interpolated_images, dtype=tf.float32)

integrated_gradients = (images[-1] - images[0]) * avg_gradients


def plot_img_attributions(integrated_gradients,
                          images,
                          cmap=None,
                          overlay_alpha=0.4):

    # Sum of the attributions across color channels for visualization.
    # The attribution mask shape is a grayscale image with height and width
    # equal to the original image.
    # attribution_mask = tf.reduce_sum(tf.math.abs(integrated_gradients), axis=-1)

    fig, axs = plt.subplots(nrows=1, ncols=4, squeeze=False, figsize=(6, 6))

    axs[0, 0].set_title('Baseline image')
    axs[0, 0].imshow(images[0])
    axs[0, 0].axis('off')

    axs[0, 1].set_title('Original image')
    axs[0, 1].imshow(images[-1])
    axs[0, 1].axis('off')

    axs[0, 2].set_title('Attribution mask')
    axs[0, 2].imshow(integrated_gradients*255, cmap=cmap)
    axs[0, 2].axis('off')

    axs[0, 3].set_title('Overlay')
    axs[0, 3].imshow(integrated_gradients*255, cmap=cmap)
    axs[0, 3].imshow(images[-1], alpha=overlay_alpha)
    axs[0, 3].axis('off')

    plt.tight_layout()
    return fig

_ = plot_img_attributions(integrated_gradients,
                          images,
                          cmap=plt.cm.inferno,
                          overlay_alpha=0.5)

