import numpy as np
import os
import cv2
import random
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import applications, losses, optimizers, metrics, Model
from tensorflow.keras.layers import Layer, Input, Dense, Flatten, Lambda, Conv2D, MaxPooling2D
from tensorflow.keras.applications import VGG16, vgg16
from matplotlib import gridspec
from matplotlib.image import imread
import matplotlib.pyplot as plt
import shutil
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import seaborn as sns
from tensorflow.keras.metrics import Recall

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

# Separate classes into train & test; executed only once


def separate_dataset(data_src):
    directories = os.listdir(data_src)
    num_directory = len(directories)
    train_test_ratio = 0.7
    n_train = int(np.ceil(num_directory * train_test_ratio))

    train_dirs = directories[0:n_train]
    test_dirs = directories[n_train:]
    if '-1' in train_dirs:
        train_dirs.remove('-1')
    if '-1' in test_dirs:
        test_dirs.remove('-1')

    for directory in os.listdir(data_src):
        old_path = os.path.join(data_src, directory)

        if directory in train_dirs:
            train_path = os.path.join(data_src, 'train', directory)
            shutil.copytree(old_path, train_path)
        elif directory in test_dirs:
            test_path = os.path.join(data_src, 'test', directory)
            shutil.copytree(old_path, test_path)


data_src = '/data/InJoon/1.5.dataset before augmentation and testing/malicious/1'
# separate_dataset(data_src)


class PreProcessing:

    images_train = np.array([])
    labels_train = np.array([])
    unique_train_label = np.array([])
    map_train_label_indices = dict()

    def __init__(self, data_src):
        self.data_src = data_src
        print("Loading the Dataset...")
        self.images_train, self.labels_train = self.preprocessing()
        self.unique_train_label = np.unique(self.labels_train)
        self.map_train_label_indices = {label: np.flatnonzero(self.labels_train == label) for label in
                                        self.unique_train_label}
        print('Preprocessing Done. Summary:')
        print("Images train :", self.images_train.shape)
        print("Labels train :", self.labels_train.shape)
        print("Unique label :", self.unique_train_label)

    def normalize(self, x):
        min_val = np.min(x)
        max_val = np.max(x)
        x = (x - min_val) / (max_val - min_val)
        return x

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

    def preprocessing(self):
        X, y = self.read_dataset()

        labels = list(set(y))
        label_dict = dict(zip(labels, range(len(labels))))
        Y = np.asarray([label_dict[label] for label in y])

        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = []
        y_shuffled = []

        for index in shuffle_indices:
            x_shuffled.append(X[index])
            y_shuffled.append(Y[index])

        return np.asarray(x_shuffled), np.asarray(y_shuffled)

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
            self.images_train[idxs_a, :])
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


class DistanceLayer(Layer):
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


dataset = PreProcessing(
    '/data/InJoon/1.5.dataset before augmentation and testing/malicious/1/train')
data = dataset.get_triplets_batch()
count = len(dataset.labels_train)

train_dataset = data.take(round(count * 0.8))
val_dataset = data.skip(round(count * 0.8))

train_dataset = train_dataset.batch(32, drop_remainder=False)
train_dataset = train_dataset.prefetch(8)
val_dataset = val_dataset.batch(32, drop_remainder=False)
val_dataset = val_dataset.prefetch(8)

model = TripletLoss()
target_shape = (224, 224)

# Setup Network
anchor_input = Input(name="anchor", shape=target_shape + (3,))
positive_input = Input(name="positive", shape=target_shape + (3,))
negative_input = Input(name="negative", shape=target_shape + (3,))

embedding = model.embedding()
distances = DistanceLayer()(
    embedding(anchor_input),
    embedding(positive_input),
    embedding(negative_input),
)

siamese_network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
)

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


# Test Dataset
def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val) / (max_val - min_val)
    return x


def read_dataset(data_src):
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
    print('Dataset loaded successfully.')
    return X, y


test_data_src = '/data/InJoon/1.5.dataset before augmentation and testing/malicious/1/test'


def preprocessing(test_data_src):
    X, y = read_dataset(test_data_src)

    labels = list(set(y))
    label_dict = dict(zip(labels, range(len(labels))))
    Y = np.asarray([label_dict[label] for label in y])

    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = []
    y_shuffled = []

    for index in shuffle_indices:
        x_shuffled.append(X[index])
        y_shuffled.append(Y[index])

    return np.asarray(x_shuffled), np.asarray(y_shuffled)


test_images, test_labels = preprocessing(test_data_src)
unique_test_labels = np.unique(test_labels)
map_test_label_indices = {label: np.flatnonzero(
    test_labels == label) for label in unique_test_labels}

# Define matrix
matrix = [[0 for x in range(922)] for y in range(922)]
rows, cols = len(matrix), len(matrix[0])

for r in range(rows):
    for c in range(r+1, cols):
        cosine_similarity = metrics.CosineSimilarity()
        im1 = tf.expand_dims(test_images[r], axis=0)
        im2 = tf.expand_dims(test_images[c], axis=0)
        embedding1 = embedding(im1)
        embedding2 = embedding(im2)

        similarity = cosine_similarity(embedding1, embedding2)

        # compute distance matrix
        matrix[r][c] = 1 - similarity.numpy()

    if r % 10 == 0:
        print("ROW:", r)

# Copy upper triagnle of the matrix to the lower triangle
matrix = np.triu(matrix)
matrix = matrix + matrix.T - np.diag(np.diag(matrix))


# Locate the optimal epsilon value where the curvature is maximum in the graph
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(matrix)
distances, indices = nbrs.kneighbors(matrix)

distances = np.sort(distances, axis=0)
distances = distances[:, 1]


fig = plt.figure()
ax = fig.gca()
ax.set_yticks(np.arange(0, 4., 0.2))
plt.grid()
plt.rcParams['figure.figsize'] = [10, 8]
plt.plot(distances)


# DBSCAN
distance_matrix = np.array(matrix)
clustering = DBSCAN(eps=0.1, min_samples=10).fit(distance_matrix)
cluster = clustering.labels_

# sort cluster and keep track of label
sorted_cluster_labels = [i[0] for i in sorted(
    enumerate(cluster), key=lambda x:x[1], reverse=True)]

# make a copy of test images and labels
test_dataset = test_images.copy()
test_labels_copy = test_labels.copy()

# rearrange test images and labels based on the sorted cluster label
test_dataset = np.array(test_dataset)
test_dataset = test_dataset[sorted_cluster_labels]
test_labels_copy = test_labels_copy[sorted_cluster_labels]

# Rearrange rows and cols on the DBSCAN cluster labels
distance_matrix = distance_matrix[:, sorted_cluster_labels]
distance_matrix = distance_matrix[sorted_cluster_labels, :]

# Plot the 2D heatmap
ax = sns.heatmap(distance_matrix, linewidth=0.01)
plt.show()

# Visualize clusters
sorted_cluster = sorted(cluster, reverse=True)
cluster_unq_list = list(set(cluster))
cluster_unq_list = sorted(cluster_unq_list, reverse=True)

for i in range(len(cluster_unq_list)):
    if i == len(cluster_unq_list)-1:
        next_label_idx = 922
    else:
        next_label_idx = sorted_cluster.index(cluster_unq_list[i+1])
    print(next_label_idx)


def show(ax, image):
    ax.imshow(image)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


fig_len = len(cluster_unq_list) * 5
fig, axs = plt.subplots(len(cluster_unq_list), 5, figsize=(100, 100))

start_idx = 0
for i in range(len(cluster_unq_list)):
    # index of the next label
    if i == len(cluster_unq_list)-1:
        next_label_idx = 922
    else:
        next_label_idx = sorted_cluster.index(cluster_unq_list[i+1])

    # choose 5 indices randomly that belong to the same label
    five_random_idx = np.random.choice(
        list(range(start_idx, next_label_idx)), 5, replace=False)
    start_idx = next_label_idx
    print(five_random_idx)

    show(axs[i, 0], test_images[five_random_idx[0]])
    show(axs[i, 1], test_images[five_random_idx[1]])
    show(axs[i, 2], test_images[five_random_idx[2]])
    show(axs[i, 3], test_images[five_random_idx[3]])
    show(axs[i, 4], test_images[five_random_idx[4]])


def generate_image_pairs(test_images):
    image_pairs = []
    labels = []

    for _ in range(len(test_labels)):
        # image pairs from the same cluster
        random_label = np.random.choice(unique_test_labels)
        a, p = np.random.choice(
            map_test_label_indices[random_label], 2, replace=False)

#        image_pairs.append((test_images[a], test_images[p]))
        image_pairs.append((a, p))
        labels.append(1)

    for _ in range(len(test_labels)):
        # image pairs from different clusters
        label_l, label_r = np.random.choice(
            unique_test_labels, 2, replace=False)
        a = np.random.choice(map_test_label_indices[label_l])
        n = np.random.choice(map_test_label_indices[label_r])

#         image_pairs.append((test_images[a], test_images[n]))
        image_pairs.append((a, n))
        labels.append(0)

    return image_pairs, labels


val_image_pairs, val_labels = generate_image_pairs(test_images)

predicted_labels = []

for idx1, idx2 in val_image_pairs:
    if cluster[idx1] == -1 and cluster[idx2] == -1:
        predicted_labels.append(-1)
    elif cluster[idx1] != cluster[idx2]:
        predicted_labels.append(0)
    else:
        predicted_labels.append(1)

m = Recall()
m.update_state(val_labels, predicted_labels)
m.result().numpy()
