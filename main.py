import numpy as np
import os
import cv2
import random
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Input, Dense, Flatten, Lambda, Conv2D, MaxPooling2D
from tensorflow.keras.applications import VGG16, vgg16
from matplotlib import gridspec
from matplotlib.image import imread
import matplotlib.pyplot as plt


# Check availability of GPU
import nvidia_smi

GPU = 0
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


class PreProcessing:

    images_train = np.array([])
    images_test = np.array([])
    labels_train = np.array([])
    labels_test = np.array([])
    anchors_train = np.array([])
    unique_train_label = np.array([])
    map_train_label_indices = dict()

    def __init__(self, data_src, data_src_anchor):
        self.data_src = data_src
        self.data_src_anchor = data_src_anchor
        print("Loading the Dataset...")
        self.anchors_train, self.images_train, self.images_test, self.labels_train, self.labels_test = self.preprocessing(
            0.9)
        self.unique_train_label = np.unique(self.labels_train)
        self.map_train_label_indices = {label: np.flatnonzero(self.labels_train == label) for label in
                                        self.unique_train_label}
        print('Preprocessing Done. Summary:')
        print("Images train :", self.images_train.shape)
        print("Labels train :", self.labels_train.shape)
        print("Images test  :", self.images_test.shape)
        print("Labels test  :", self.labels_test.shape)
        print("Unique label :", self.unique_train_label)

    def normalize(self, x):
        min_val = np.min(x)
        max_val = np.max(x)
        x = (x - min_val) / (max_val - min_val)
        return x

    def read_anchor_dataset(self):
        count = 0
        for directory in os.listdir(self.data_src_anchor):
            count += len([file for file in os.listdir(os.path.join(self.data_src_anchor, directory))])

        X = [None] * count
        y = [None] * count
        idx = 0

        for directory in os.listdir(self.data_src_anchor):
            try:
                print('Read directory: ', directory)
                for pic in os.listdir(os.path.join(self.data_src_anchor, directory)):
                    img = imread(os.path.join(
                        self.data_src_anchor, directory, pic))
                    img = tf.image.resize(img, (224, 224))
                    img = self.normalize(img)

                    X[idx] = np.squeeze(np.asarray(img))
                    y[idx] = directory
                    idx += 1

            except Exception as e:
                print('Failed to read images from Directory: ', directory)
                print('Exception Message: ', e)
        print('Dataset loaded successfully.')
        return X, y

    def read_dataset(self):
        count = 0
        for directory in os.listdir(self.data_src):
            count += len([file for file in os.listdir(os.path.join(self.data_src, directory))])

        X = [None] * count
        y = [None] * count
        idx = 0

        for directory in os.listdir(self.data_src):
            try:
                print('Read directory: ', directory)
                for pic in os.listdir(os.path.join(self.data_src, directory)):
                    img = imread(os.path.join(self.data_src, directory, pic))
                    img = tf.image.resize(img, (224, 224))
                    img = self.normalize(img)

                    X[idx] = np.squeeze(np.asarray(img))
                    y[idx] = directory
                    idx += 1

            except Exception as e:
                print('Failed to read images from Directory: ', directory)
                print('Exception Message: ', e)
        print('Dataset loaded successfully.')
        return X, y

    def preprocessing(self, train_test_ratio):
        X, y = self.read_dataset()
        X_a, y_a = self.read_anchor_dataset()

        labels = list(set(y))
        label_dict = dict(zip(labels, range(len(labels))))
        Y = np.asarray([label_dict[label] for label in y])

        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = []
        y_shuffled = []
        a_shuffled = []

        for index in shuffle_indices:
            x_shuffled.append(X[index])
            y_shuffled.append(Y[index])

            a_index = random.choice(
                [i for i, a in enumerate(y_a) if label_dict[a] == Y[index]])
            a_shuffled.append(X_a[a_index])

        size_of_dataset = len(x_shuffled)
        n_train = int(np.ceil(size_of_dataset * train_test_ratio))

        return np.asarray(a_shuffled), np.asarray(x_shuffled[0:n_train]), np.asarray(x_shuffled[n_train + 1:size_of_dataset]), np.asarray(
            y_shuffled[0:n_train]), np.asarray(y_shuffled[n_train + 1:size_of_dataset])

    def get_triplets(self):
        label_l, label_r = np.random.choice(
            self.unique_train_label, 2, replace=False)
        a, p = np.random.choice(
            self.map_train_label_indices[label_l], 2, replace=False)
        n = np.random.choice(self.map_train_label_indices[label_r])
        return a, p, n

    def get_triplets_batch(self):
        idxs_a, idxs_p, idxs_n = [], [], []
        n = len(self.labels_train)
        for _ in range(n):
            a, p, n = self.get_triplets()
            idxs_a.append(a)
            idxs_p.append(p)
            idxs_n.append(n)

        anchor_dataset = tf.data.Dataset.from_tensor_slices(
            self.anchors_train[idxs_a, :])
        positive_dataset = tf.data.Dataset.from_tensor_slices(
            self.images_train[idxs_p, :])
        negative_dataset = tf.data.Dataset.from_tensor_slices(
            self.images_train[idxs_n, :])

        dataset = tf.data.Dataset.zip(
            (anchor_dataset, positive_dataset, negative_dataset))

        return dataset


class TripletLoss:

    def embedding(self):
        inp = Input(shape=(224, 224, 3))

        # first block
        c1 = Conv2D(32, (7, 7), activation='relu', padding='same',
                    kernel_initializer=tf.keras.initializers.GlorotNormal())(inp)
        m1 = MaxPooling2D((2, 2), padding='same')(c1)

        # second block
        c2 = Conv2D(64, (5, 5), activation='relu', padding='same',
                    kernel_initializer=tf.keras.initializers.GlorotNormal())(m1)
        m2 = MaxPooling2D((2, 2), padding='same')(c2)

        # third block
        c3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                    kernel_initializer=tf.keras.initializers.GlorotNormal())(m2)
        m3 = MaxPooling2D((2, 2), padding='same')(c3)

        # fourth block
        c4 = Conv2D(256, (1, 1), activation='relu', padding='same',
                    kernel_initializer=tf.keras.initializers.GlorotNormal())(m3)
        m4 = MaxPooling2D((2, 2), padding='same')(c4)

        c5 = Conv2D(28, (1, 1), activation=None, padding='same',
                    kernel_initializer=tf.keras.initializers.GlorotNormal())(m4)
        m5 = MaxPooling2D((2, 2), padding='same')(c5)

        f1 = Flatten()(m5)

        return Model(inputs=[inp], outputs=[f1], name='embedding')


class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):

        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):

        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # List metrics here so the `reset_states()` can be called automatically.
        return [self.loss_tracker]


dataset = PreProcessing('./after_augmentation/', './before_augmentation/')

model = TripletLoss()
target_shape = (224, 224)

# Setup Network
anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
positive_input = layers.Input(name="positive", shape=target_shape + (3,))
negative_input = layers.Input(name="negative", shape=target_shape + (3,))

embedding = model.embedding()
distances = DistanceLayer()(
    embedding(anchor_input),
    embedding(positive_input),
    embedding(negative_input),
)

siamese_network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
)

data = dataset.get_triplets_batch()
count = len(dataset.labels_train)
train_dataset = data.take(round(count * 0.8))
val_dataset = data.skip(round(count * 0.8))

train_dataset = train_dataset.batch(32, drop_remainder=False)
train_dataset = train_dataset.prefetch(8)

val_dataset = val_dataset.batch(32, drop_remainder=False)
val_dataset = val_dataset.prefetch(8)

siamese_model = SiameseModel(siamese_network)
siamese_model.compile(optimizer=optimizers.Adam(0.0001), metrics=['accuracy'])
siamese_model.fit(train_dataset, epochs=20, validation_data=val_dataset)

sample = train_dataset.as_numpy_iterator().next()

anchor, positive, negative = sample
anchor_embedding, positive_embedding, negative_embedding = (
    embedding(anchor),
    embedding(positive),
    embedding(negative),
)


def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val) / (max_val - min_val)
    return x


def get_train_test_dataset(train_test_ratio):
    data_src = './after_augmentation/'

    count = 0
    for directory in os.listdir(data_src):
        count += len([file for file in os.listdir(os.path.join(data_src, directory))])

    X = [None] * count
    y = [None] * count
    idx = 0

    for directory in os.listdir(data_src):
        try:
            print('Read directory: ', directory)
            for pic in os.listdir(os.path.join(data_src, directory)):
                img = imread(os.path.join(data_src, directory, pic))
                img = tf.image.resize(img, (224, 224))
                img = normalize(img)

                X[idx] = np.squeeze(np.asarray(img))
                y[idx] = directory
                idx += 1

        except Exception as e:
            print('Failed to read images from Directory: ', directory)
            print('Exception Message: ', e)

    labels = list(set(y))
    label_dict = dict(zip(labels, range(len(labels))))
    Y = np.asarray([label_dict[label] for label in y])
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = []
    y_shuffled = []
    for index in shuffle_indices:
        x_shuffled.append(X[index])
        y_shuffled.append(Y[index])

    size_of_dataset = len(x_shuffled)
    n_train = int(np.ceil(size_of_dataset * train_test_ratio))
    n_test = int(np.ceil(size_of_dataset * (1 - train_test_ratio)))
    return np.asarray(x_shuffled[0:n_train]), np.asarray(x_shuffled[n_train + 1:size_of_dataset]), np.asarray(y_shuffled[0:n_train]), np.asarray(y_shuffled[
        n_train + 1:size_of_dataset])


train_images, test_images, train_label, test_label = get_train_test_dataset(
    0.7)
