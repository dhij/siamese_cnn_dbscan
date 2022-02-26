import os
import pathlib
import cv2

import numpy as np
import PIL.Image as Image
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras
from tensorflow.keras.layers import Layer, Input, Dense, Flatten

from sklearn.model_selection import train_test_split

import random
import shutil

# Check availability of GPU
import nvidia_smi
import os

GPU = 1
os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(GPU)

gpus = tf.config.experimental.list_physical_devices('GPU')
print('GPU is', 'available' if gpus else 'NOT AVAILABLE')
print('Number GPUs Available: ', len(gpus))


# Restrict TensorFlow to only use the MENTIONED GPU
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print('{} Physical GPUs, Logical GPU'.format(
            len(gpus), len(logical_gpus)))
        print('Memory growth set to: {} '.format(
            str(tf.config.experimental.get_memory_growth(gpus[0]))))
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

# Initiate NVIDIA-SMI
nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(GPU)


# Get GPU memory usage
def print_GPU_usage():
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print("Total memory: {}".format(info.total // (1024 * 1024)))
    print("Used memory: {}".format(info.used // (1024 * 1024)))


# Set up directories

# for each of the dirs, create anchor, positive, & negative
for directory in os.listdir('before_augmentation'):
    POS_PATH = os.path.join('siamese_data', directory, 'positive')
    NEG_PATH = os.path.join('siamese_data', directory, 'negative')
    ANC_PATH = os.path.join('siamese_data', directory, 'anchor')
    os.makedirs(POS_PATH)
    os.makedirs(NEG_PATH)
    os.makedirs(ANC_PATH)

# copy images to each of anchor directories
for directory in os.listdir('before_augmentation'):
    for file in os.listdir(os.path.join('before_augmentation', directory)):
        EX_PATH = os.path.join('before_augmentation', directory, file)
        NEW_PATH = os.path.join('siamese_data', directory, 'anchor', file)
        shutil.copy(EX_PATH, NEW_PATH)


# copy images to positive directory in the corresponding cluster
for directory in os.listdir('after_augmentation'):
    for file in os.listdir(os.path.join('after_augmentation', directory)):
        EX_PATH = os.path.join('after_augmentation', directory, file)
        NEW_PATH = os.path.join('siamese_data', directory, 'positive', file)
        shutil.copy(EX_PATH, NEW_PATH)

# randomly select 30 images from other clusters and move them to negative directory
for directory in os.listdir('siamese_data'):
    for other_directory in os.listdir('siamese_data'):
        if other_directory != directory:
            samples = random.sample(os.listdir(os.path.join(
                'siamese_data', other_directory, 'positive')), 30)
            for file in samples:
                EX_PATH = os.path.join(
                    'siamese_data', other_directory, 'positive', file)
                NEW_PATH = os.path.join(
                    'siamese_data', directory, 'negative', file)
                shutil.copy(EX_PATH, NEW_PATH)


def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (224, 224))
    img = img / 255.0
    return img


arr_len = len([f for f in os.listdir('siamese_data')])
num_files = []
anchors, positives, negatives = [None] * \
    arr_len, [None] * arr_len, [None] * arr_len
index = 0

# determine number of files in each anchor directory (max number of files we can take)
for directory in sorted(os.listdir('siamese_data')):
    num_files.append(
        len([f for f in os.listdir(os.path.join('siamese_data', directory, 'anchor'))]))


# store in array of anchor/positive/negative datasets
for directory in sorted(os.listdir('siamese_data')):
    numFiles = 300 if num_files[index] > 300 else num_files[index]
    anchors[index] = tf.data.Dataset.list_files(os.path.join(
        'siamese_data', directory, 'anchor') + '/*.jpg').take(numFiles)
    positives[index] = tf.data.Dataset.list_files(os.path.join(
        'siamese_data', directory, 'positive') + '/*.jpg').take(numFiles)
    negatives[index] = tf.data.Dataset.list_files(os.path.join(
        'siamese_data', directory, 'negative') + '/*.jpg').take(numFiles)
    index += 1

data = [None] * arr_len
index = 0

for _ in range(arr_len):
    positive_temp = tf.data.Dataset.zip(
        (anchors[index], positives[index], tf.data.Dataset.from_tensor_slices(tf.ones(len(anchors[index])))))
    negative_temp = tf.data.Dataset.zip(
        (anchors[index], negatives[index], tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchors[index])))))
    data[index] = positive_temp.concatenate(negative_temp)
    index += 1

# 0-26 dirs for test_data and 27-32 dirs for train_data
test_data_temp = data[len(data) - 6:]
train_data_temp = data[:len(data) - 6]


def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)


# combine array of train_data into a single dataset
train_data = train_data_temp[0]
for i in range(1, len(train_data_temp)):
    train_data = train_data.concatenate(train_data_temp[i])

# build dataloader pipeline
train_data = train_data.map(preprocess_twin)
train_data = train_data.cache()
train_data = train_data.shuffle(buffer_size=1024)

train_data = train_data.batch(16)
train_data = train_data.prefetch(8)


def make_base_model(inputs):
    base_model = keras.applications.VGG16(
        weights='imagenet',
        input_shape=(224, 224, 3),
        include_top=False
    )

    base_model.trainable = False
    x = base_model(inputs, training=False)

    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = keras.layers.GlobalAveragePooling2D()(x)

    # f_layer = Flatten()(x)
    outputs = Dense(4096, activation='sigmoid')(x)

    return keras.Model(inputs, outputs)

# Siamese L1 Distance class


class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


def make_siamese_model():

    # Anchor image input
    inp_image = Input(name='input_img', shape=(224, 224, 3))

    # Validation image input
    val_image = Input(name='validation_img', shape=(224, 224, 3))

    siamese_layer = L1Dist()
    inp_embedding = make_base_model(inp_image)
    val_embedding = make_base_model(val_image)

    distances = siamese_layer(inp_embedding(
        inp_image), val_embedding(val_image))

    classifier = Dense(1, activation='sigmoid')(distances)

    return keras.Model(inputs=[inp_image, val_image], outputs=classifier, name='SiaemeseNetwork')


siamese_model = make_siamese_model()

# Training
binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)


@tf.function
def train_step(batch):

    # record all operations
    with tf.GradientTape() as tape:
        # get anchor and positive/negative image
        X = batch[:2]
        # get the label
        y = batch[2]

        # forward pass
        yhat = siamese_model(X, training=True)
        # calculate loss
        loss = binary_cross_loss(y, yhat)
    print(loss)

    # calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)

    # calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    return loss


def train(data, EPOCHS):
    # loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))

        # loop through each batch
        for idx, batch in enumerate(data):
            # run train step
            train_step(batch)
            progbar.update(idx+1)

        # save checkpoints
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


EPOCHS = 50

train(train_data, EPOCHS)
