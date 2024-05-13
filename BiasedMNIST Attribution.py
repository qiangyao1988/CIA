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


### IG

baseline = tf.zeros(shape=(28,28,3))
m_steps= 10
alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps) # Generate m_steps intervals for integral_approximation() below.


def interpolate_images(baseline, image, alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x +  alphas_x * delta
    return images

interpolated_images = interpolate_images(baseline=baseline, image=x_te[1], alphas=alphas)

def compute_gradients(images, target_class_idx):
    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = model(images)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    return tape.gradient(probs, images)

path_gradients = compute_gradients(
    images=interpolated_images,
    target_class_idx=1)

pred = model(interpolated_images)
pred_proba = tf.nn.softmax(pred, axis=-1)[:, 0]

plt.figure(figsize=(10, 4))
ax1 = plt.subplot(1, 2, 1)
ax1.plot(alphas, pred_proba)
ax1.set_title('Target class predicted probability over alpha')
ax1.set_ylabel('model p(target class)')
ax1.set_xlabel('alpha')
ax1.set_ylim([0, 1])

ax2 = plt.subplot(1, 2, 2)
# Average across interpolation steps
average_grads = tf.reduce_mean(path_gradients, axis=[1, 2, 3])
# Normalize gradients to 0 to 1 scale. E.g. (x - min(x))/(max(x)-min(x))
average_grads_norm = (average_grads-tf.math.reduce_min(average_grads))/(tf.math.reduce_max(average_grads)-tf.reduce_min(average_grads))
ax2.plot(alphas, average_grads_norm)
ax2.set_title('Average pixel gradients (normalized) over alpha')
ax2.set_ylabel('Average pixel gradients')
ax2.set_xlabel('alpha')
ax2.set_ylim([0, 1]);


def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients


@tf.function
def integrated_gradients(baseline, image,  target_class_idx, m_steps=50, batch_size=32):
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
    # integrated_gradients =  avg_gradients
    return integrated_gradients
    

def plot_img_attributions(baseline, image, target_class_idx, m_steps=50, cmap=None, overlay_alpha=0.4):

    attributions = integrated_gradients(baseline=baseline,
                                      image=image,
                                      target_class_idx=target_class_idx,
                                      m_steps=m_steps)

    # Sum of the attributions across color channels for visualization.
    # The attribution mask shape is a grayscale image with height and width
    # equal to the original image.
    attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)

    fig, axs = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(5, 5))

    axs[0, 0].set_title('Baseline image')
    axs[0, 0].imshow(baseline)
    axs[0, 0].axis('off')

    axs[0, 1].set_title('Original image')
    axs[0, 1].imshow(image)
    axs[0, 1].axis('off')
    
    
    axs[0, 2].set_title('Attribution mask')
    axs[0, 2].imshow(attribution_mask, cmap=cmap)
    axs[0, 2].axis('off')

    plt.tight_layout()
    return fig


_ = plot_img_attributions(image=x_te[0], baseline=baseline, target_class_idx=1, m_steps=10, cmap=plt.cm.inferno, overlay_alpha=0.5)


def plot_img_attributions(baseline, image, target_class_idx, m_steps=50, cmap=None, overlay_alpha=0.4):

    attributions = integrated_gradients(baseline=baseline,
                                      image=image,
                                      target_class_idx=target_class_idx,
                                      m_steps=m_steps)

    # Sum of the attributions across color channels for visualization.
    # The attribution mask shape is a grayscale image with height and width
    # equal to the original image.
    # attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)

    fig, axs = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(5, 5))

    axs[0, 0].set_title('Baseline image')
    axs[0, 0].imshow(baseline)
    axs[0, 0].axis('off')

    axs[0, 1].set_title('Original image')
    axs[0, 1].imshow(image)
    axs[0, 1].axis('off')
    
    
    axs[0, 2].set_title('Attribution mask')
    axs[0, 2].imshow(attributions*255., cmap=cmap)
    axs[0, 2].axis('off')

    plt.tight_layout()
    return fig


_ = plot_img_attributions(image=x_te[0], baseline=baseline, target_class_idx=1, m_steps=10, cmap=plt.cm.inferno, overlay_alpha=0.5)


def generate_examples(image_batch, label_batch):
    images = []
    lables = []
    attributions = []
    
    baseline = tf.zeros(shape=(28,28,3))
    
    for image,label in zip(image_batch, label_batch):

        attribution = integrated_gradients(baseline=baseline,
                                  image=image,
                                  target_class_idx=label,
                                  m_steps=10)
        images.append(image)
        lables.append(label)
        attributions.append(attribution*255.)
        
    return (np.asarray(images),np.asarray(lables),np.asarray(attributions))


images,lables,attributions = generate_examples(image_batch, label_batch)


def single_run_deletion(image,attribution,model):
    
    start = np.copy(image)
    finish = np.zeros(shape=(28,28,3))
    
    HW = 784
    n_steps = (HW + 28 - 1) // 28
    scores = np.empty(n_steps + 1)
    
    salient_order = np.flip(np.argsort(attribution.reshape(-1, HW), axis=1), axis=-1)
    
    pred = model.predict(tf.expand_dims(image, 0))
    score = tf.nn.softmax(pred)
    top = np.argmax(score, -1)
    
    for i in range(n_steps+1):
        pred = model.predict(tf.expand_dims(start, 0))
        score = tf.nn.softmax(pred)
        scores[i] = score[0, top]
        if i < n_steps:
            coords = salient_order[:, 28 * i:28 * (i + 1)]
            start.reshape(1, 3, HW)[0, :, coords] = finish.reshape(1, 3, HW)[0, :, coords]
            
    return scores


def get_deletion_score(images,attributions,model):
    scores = []
    for image,attribution in zip(images, attributions):
        score = single_run_deletion(image,attribution,model).mean()
        scores.append(score)
    return scores


def single_run_insertion(image,attribution,model):
    
    finish = np.copy(image)
    start = np.zeros(shape=(28,28,3))
    
    HW = 784
    n_steps = (HW + 28 - 1) // 28
    scores = np.empty(n_steps + 1)
    
    salient_order = np.flip(np.argsort(attribution.reshape(-1, HW), axis=1), axis=-1)
    
    pred = model.predict(tf.expand_dims(image, 0))
    score = tf.nn.softmax(pred)
    top = np.argmax(score, -1)
    
    for i in range(n_steps+1):
        pred = model.predict(tf.expand_dims(start, 0))
        score = tf.nn.softmax(pred)
        scores[i] = score[0, top]
        if i < n_steps:
            coords = salient_order[:, 28 * i:28 * (i + 1)]
            start.reshape(1, 3, HW)[0, :, coords] = finish.reshape(1, 3, HW)[0, :, coords]
            
    return scores

def get_insertion_score(images,attributions,model):
    scores = []
    for image,attribution in zip(images, attributions):
        score = single_run_insertion(image,attribution,model).mean()
        scores.append(score)
    return scores


### CGI

m_steps=10
alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps)


def interpolate_images(image, color, alphas):
    
    images = []   
    image = np.tile(image[:, :, np.newaxis], 3)
    
    for alpha in alphas:
        inter_red = (1-alpha) * red + alpha * green
        inter_green = (1-alpha) * green + alpha * red
        
        aug_image = np.zeros((28, 28, 3), dtype='float32')
        
        if color: 
            aug_image[:, :, 0] = image[:, :, 0] / 255 * inter_green[0]
            aug_image[:, :, 1] = image[:, :, 1] / 255 * inter_green[1]
            aug_image[:, :, 2] = image[:, :, 2] / 255 * inter_green[2]
        else:
            aug_image[:, :, 0] = image[:, :, 0] / 255 * inter_red[0]
            aug_image[:, :, 1] = image[:, :, 1] / 255 * inter_red[1]
            aug_image[:, :, 2] = image[:, :, 2] / 255 * inter_red[2]
        
        aug_image = aug_image/255.
        images.append(aug_image)
        
    return np.asarray(images)

interpolated_images = interpolate_images(x_test[1], c_te[1], alphas=alphas)


def compute_gradients(images, target_class_idx):
    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = model(images)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    return tape.gradient(probs, images)


path_gradients = compute_gradients(
    images=tf.convert_to_tensor(interpolated_images, dtype=tf.float32),
    target_class_idx=1)


pred = model(interpolated_images)
pred_proba = tf.nn.softmax(pred, axis=-1)[:, 0]

plt.figure(figsize=(10, 4))
ax1 = plt.subplot(1, 2, 1)
ax1.plot(alphas, pred_proba)
ax1.set_title('Target class predicted probability over alpha')
ax1.set_ylabel('model p(target class)')
ax1.set_xlabel('alpha')
ax1.set_ylim([0, 1])

ax2 = plt.subplot(1, 2, 2)
# Average across interpolation steps
average_grads = tf.reduce_mean(path_gradients, axis=[1, 2, 3])
# Normalize gradients to 0 to 1 scale. E.g. (x - min(x))/(max(x)-min(x))
average_grads_norm = (average_grads-tf.math.reduce_min(average_grads))/(tf.math.reduce_max(average_grads)-tf.reduce_min(average_grads))
ax2.plot(alphas, average_grads_norm)
ax2.set_title('Average pixel gradients (normalized) over alpha')
ax2.set_ylabel('Average pixel gradients')
ax2.set_xlabel('alpha')
ax2.set_ylim([0, 1]);

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
    attribution_mask = tf.reduce_sum(tf.math.abs(integrated_gradients), axis=-1)

    fig, axs = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(5, 5))

    axs[0, 0].set_title('Baseline image')
    axs[0, 0].imshow(images[0])
    axs[0, 0].axis('off')

    axs[0, 1].set_title('Original image')
    axs[0, 1].imshow(images[-1])
    axs[0, 1].axis('off')

    axs[0, 2].set_title('Attribution mask')
    axs[0, 2].imshow(attribution_mask, cmap=cmap)
    axs[0, 2].axis('off')

    #     axs[1, 1].set_title('Overlay')
    #     axs[1, 1].imshow(attribution_mask, cmap=cmap)
    #     axs[1, 1].imshow(images[-1], alpha=overlay_alpha)
    #     axs[1, 1].axis('off')

    plt.tight_layout()
    return fig

_ = plot_img_attributions(integrated_gradients,
                          images,
                          cmap=plt.cm.inferno,
                          overlay_alpha=0.5)


def plot_img_attributions(integrated_gradients,
                          images,
                          cmap=None,
                          overlay_alpha=0.4):

    # Sum of the attributions across color channels for visualization.
    # The attribution mask shape is a grayscale image with height and width
    # equal to the original image.
    # attribution_mask = tf.reduce_sum(tf.math.abs(integrated_gradients), axis=-1)

    fig, axs = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(5, 5))

    axs[0, 0].set_title('Baseline image')
    axs[0, 0].imshow(images[0])
    axs[0, 0].axis('off')

    axs[0, 1].set_title('Original image')
    axs[0, 1].imshow(images[-1])
    axs[0, 1].axis('off')

    axs[0, 2].set_title('Attribution mask')
    axs[0, 2].imshow(integrated_gradients*255., cmap=cmap)
    axs[0, 2].axis('off')

    plt.tight_layout()
    return fig

_ = plot_img_attributions(integrated_gradients,
                          images,
                          cmap=plt.cm.inferno,
                          overlay_alpha=0.5)


def CGI(image, label, color):
    m_steps=10
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps)
    interpolated_images = interpolate_images(image, color, alphas=alphas)
    path_gradients = compute_gradients(images=tf.convert_to_tensor(interpolated_images, dtype=tf.float32), 
                                       target_class_idx=label)
    avg_gradients = integral_approximation(gradients=path_gradients)
    images=tf.convert_to_tensor(interpolated_images, dtype=tf.float32)
    integrated_gradients = (images[-1] - images[0]) * avg_gradients
    return integrated_gradients*255.

def generate_examples(original_image_batch, image_batch, label_batch, color_batch):
    
    images = []
    lables = []
    attributions = []

    for original_image,image,label,color in zip(original_image_batch, image_batch, label_batch, color_batch):

        attribution = CGI(original_image, label, color)
        
        images.append(image)
        lables.append(label)
        attributions.append(attribution)
        
    return (np.asarray(images),np.asarray(lables),np.asarray(attributions))


### BlurIG

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


interpolated_images = interpolate_images(image=x_te[1])


pred = model(np.asarray(interpolated_images))
pred_proba = tf.nn.softmax(pred, axis=-1)[:, 0]

plt.figure(figsize=(10, 4))
ax1 = plt.subplot(1, 2, 1)
ax1.plot(alphas, pred_proba)
ax1.set_title('Target class predicted probability over alpha')
ax1.set_ylabel('model p(target class)')
ax1.set_xlabel('alpha')
ax1.set_ylim([0, 1])

pred = model(np.asarray(interpolated_images))
pred_proba = tf.nn.softmax(pred, axis=-1)[:, 0]

plt.figure(figsize=(10, 4))
ax1 = plt.subplot(1, 2, 1)
ax1.plot(alphas, pred_proba)
ax1.set_title('Target class predicted probability over alpha')
ax1.set_ylabel('model p(target class)')
ax1.set_xlabel('alpha')
ax1.set_ylim([0, 1])

ax2 = plt.subplot(1, 2, 2)
# Average across interpolation steps
average_grads = tf.reduce_mean(path_gradients, axis=[1, 2, 3])
# Normalize gradients to 0 to 1 scale. E.g. (x - min(x))/(max(x)-min(x))
average_grads_norm = (average_grads-tf.math.reduce_min(average_grads))/(tf.math.reduce_max(average_grads)-tf.reduce_min(average_grads))
ax2.plot(alphas, average_grads_norm)
ax2.set_title('Average pixel gradients (normalized) over alpha')
ax2.set_ylabel('Average pixel gradients')
ax2.set_xlabel('alpha')
ax2.set_ylim([0, 1]);


def compute_gradients(images, target_class_idx):
    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = model(images)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    return tape.gradient(probs, images)


path_gradients = compute_gradients(
    images=tf.convert_to_tensor(interpolated_images, dtype=tf.float32),
    target_class_idx=0)


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
    attribution_mask = tf.reduce_sum(tf.math.abs(integrated_gradients), axis=-1)

    fig, axs = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(5, 5))

    axs[0, 0].set_title('Baseline image')
    axs[0, 0].imshow(images[0])
    axs[0, 0].axis('off')

    axs[0, 1].set_title('Original image')
    axs[0, 1].imshow(images[-1])
    axs[0, 1].axis('off')

    axs[0, 2].set_title('Attribution mask')
    axs[0, 2].imshow(attribution_mask, cmap=cmap)
    axs[0, 2].axis('off')

    plt.tight_layout()
    return fig

_ = plot_img_attributions(integrated_gradients,
                          images,
                          cmap=plt.cm.inferno,
                          overlay_alpha=0.5)


def plot_img_attributions(integrated_gradients,
                          images,
                          cmap=None,
                          overlay_alpha=0.4):

    # Sum of the attributions across color channels for visualization.
    # The attribution mask shape is a grayscale image with height and width
    # equal to the original image.

    fig, axs = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(5, 5))

    axs[0, 0].set_title('Baseline image')
    axs[0, 0].imshow(images[0])
    axs[0, 0].axis('off')

    axs[0, 1].set_title('Original image')
    axs[0, 1].imshow(images[-1])
    axs[0, 1].axis('off')

    axs[0, 2].set_title('Attribution mask')
    axs[0, 2].imshow(integrated_gradients*255., cmap=cmap)
    axs[0, 2].axis('off')

    plt.tight_layout()
    return fig

_ = plot_img_attributions(integrated_gradients,
                          images,
                          cmap=plt.cm.inferno,
                          overlay_alpha=0.5)


def Blur_IG(image, label):
    interpolated_images = interpolate_images(image)
    path_gradients = compute_gradients(images=tf.convert_to_tensor(interpolated_images, dtype=tf.float32),
                                       target_class_idx=label)
    avg_gradients = integral_approximation(gradients=path_gradients)
    images=tf.convert_to_tensor(interpolated_images, dtype=tf.float32)
    integrated_gradients = (images[-1] - images[0]) * avg_gradients
    return integrated_gradients*255.

def generate_examples(image_batch, label_batch):
    
    images = []
    lables = []
    attributions = []

    for image,label in zip(image_batch, label_batch):

        attribution = Blur_IG(image, label)
        
        images.append(image)
        lables.append(label)
        attributions.append(attribution)
        
    return (np.asarray(images),np.asarray(lables),np.asarray(attributions))