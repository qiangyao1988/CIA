#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.losses import binary_crossentropy, mean_squared_error
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Lambda, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from scipy.stats import norm


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
(x_tr,y_tr),(x_te,y_te)=mnist.load_data()
x_tr, y_tr, c_tr = add_color(x_tr, y_tr, e=0.2)
x_te, y_te, c_te = add_color(x_te, y_te, e=0.9)
x_tr, x_te = x_tr.astype('float32')/255., x_te.astype('float32')/255.

x_tr, x_te = x_tr.reshape(x_tr.shape[0], -1), x_te.reshape(x_te.shape[0], -1)
# print(x_tr.shape, x_te.shape)

# one-hot encoding
y_tr, y_te = to_categorical(y_tr), to_categorical(y_te)
# print(y_tr.shape, y_te.shape)


# one-hot encoding
c_tr, c_te = to_categorical(c_tr), to_categorical(c_te)
# print(c_tr.shape, c_te.shape)

# network parameters
batch_size, n_epoch = 64, 50
n_hidden, z_dim = 512, 2

# encoder
x = Input(shape=(x_tr.shape[1:]))
l_condition = Input(shape=(y_tr.shape[1],))
c_condition = Input(shape=(c_tr.shape[1],))

inputs = concatenate([x, l_condition, c_condition])
x_encoded = Dense(n_hidden, activation='relu')(inputs)
x_encoded = Dense(n_hidden//2, activation='relu')(x_encoded)
mu = Dense(z_dim, activation='linear')(x_encoded)
log_var = Dense(z_dim, activation='linear')(x_encoded)

# sampling function
def sampling(args):
    mu, log_var = args
    eps = K.random_normal(shape=(batch_size, z_dim), mean=0., stddev=1.0)
    return mu + K.exp(log_var/2.) * eps

z = Lambda(sampling, output_shape=(z_dim,))([mu, log_var])
z_cond = concatenate([z, l_condition, c_condition])

# decoder
z_decoder1 = Dense(n_hidden//2, activation='relu')
z_decoder2 = Dense(n_hidden, activation='relu')
y_decoder = Dense(x_tr.shape[1], activation='sigmoid')

z_decoded = z_decoder1(z_cond)
z_decoded = z_decoder2(z_decoded)
y = y_decoder(z_decoded)

# loss
reconstruction_loss = binary_crossentropy(x, y) * x_tr.shape[1]
kl_loss = 0.5 * K.sum(K.square(mu) + K.exp(log_var) - log_var - 1, axis = -1)
cvae_loss = reconstruction_loss + kl_loss

# build model
cvae = Model([x, l_condition, c_condition], y)
cvae.add_loss(cvae_loss)
cvae.compile(optimizer='adam')
cvae.summary()

# train
cvae.fit([x_tr, y_tr, c_tr],
       shuffle=True,
       epochs=n_epoch,
       batch_size=batch_size,
       validation_data=([x_te, y_te, c_te], None), verbose=1)

# build encoder
encoder = Model([x, l_condition, c_condition], mu)
encoder.summary()

# build decoder
decoder_input = Input(shape=(z_dim+y_tr.shape[1]+c_tr.shape[1]))
_z_decoded = z_decoder1(decoder_input)
_z_decoded = z_decoder2(_z_decoded)
_y = y_decoder(_z_decoded)
generator = Model(decoder_input, _y)
generator.summary()

# exploring the latent space: change z sample on the x-axis
digit_size = 28
for cond_num in range(2):
    l_condition_num = to_categorical(cond_num, 2).reshape(1,-1)
    c_condition_num = to_categorical(cond_num, 2).reshape(1,-1)
    plt.figure(figsize=(20, 2))

    for i in range(10):
        z_sample = np.array([[0.3*i, 0.3]])
        x_decoded = generator.predict(np.column_stack([z_sample, l_condition_num, c_condition_num]))
        digit = x_decoded[0].reshape(digit_size, digit_size, 3)

        plt.subplot(1, 10, i+1)
        plt.axis('off')
        plt.imshow(digit, cmap='Greys_r',)

plt.show()

# exploring the latent space: change z sample on the y-axis
digit_size = 28
for cond_num in range(2):
    l_condition_num = to_categorical(cond_num, 2).reshape(1,-1)
    c_condition_num = to_categorical(cond_num, 2).reshape(1,-1)
    plt.figure(figsize=(20, 2))

    for i in range(10):
        z_sample = np.array([[0.3, 0.3*i]])
        x_decoded = generator.predict(np.column_stack([z_sample, l_condition_num, c_condition_num]))
        digit = x_decoded[0].reshape(digit_size, digit_size, 3)

        plt.subplot(1, 10, i+1)
        plt.axis('off')
        plt.imshow(digit, cmap='Greys_r',)

plt.show()