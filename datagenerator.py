import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from config import *


class Minc2500DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, db_dir, batch_size, input_shape, num_classes,
                 shuffle=True, img_limit=None, test_count=100):
        # TODO your initialization
        # you might want to store the parameters into class variables
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.img_limit = img_limit
        self.test_count = test_count
        # load the data from the root directory
        self.data, self.labels = self.get_data(db_dir)
        self.test_images, self.test_labels = self.get_test_data(db_dir)
        self.indices = np.arange(len(self.data))
        self.on_epoch_end()

    def get_data(self, root_dir):
        """"
        Loads the paths to the images and their corresponding labels from the database directory
        """
        # TODO your code here
        self.data = []
        self.labels = []
        for class_index in range(len(CLASSES)):
            for i in os.listdir(f'{root_dir}/minc-2500/images/{CLASSES[class_index]}')[:self.img_limit]:
                self.data.append(f'{root_dir}/minc-2500/images/{CLASSES[class_index]}/{i}')
                self.labels.append(class_index)
                # print(f'{self.data[-1]}:{self.labels[-1]}')

        return self.data, self.labels

    def get_test_data(self, root_dir):
        self.test_images = []
        self.test_labels = []
        for class_index in range(len(CLASSES)):
            for i in os.listdir(f'{root_dir}/minc-2500/images/{CLASSES[class_index]}')[-self.test_count:]:
                self.test_images.append(f'{root_dir}/minc-2500/images/{CLASSES[class_index]}/{i}')
                self.test_labels.append(class_index)
                # print(f'{self.data[-1]}:{self.labels[-1]}')

        self.test_images = self.test_images

        return self.test_images, self.test_labels

    def test_data(self):
        images = []
        labels = []

        for index in range(len(self.test_images)):
            img = cv2.imread(self.test_images[index])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))

            images.append(img)
            labels.append(self.labels[index])

        test_x = np.asarray(images) / 255.0
        test_y = np.asarray(labels)
        # optionally you can use: batch_y = tf.keras.utils.to_categorical(batch_y, num_classes=self.num_classes)
        return test_x, test_y

    def __len__(self):
        """
        Returns the number of batches per epoch: the total size of the dataset divided by the batch size
        """
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        """"
        Generates a batch of data
        """
        batch_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        images = []
        labels = []

        for index in batch_indices:
                img = cv2.imread(self.data[index])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))

                images.append(img)
                labels.append(self.labels[index])

        batch_x = np.asarray(images) / 255.0
        batch_y = np.asarray(labels)
        # optionally you can use: batch_y = tf.keras.utils.to_categorical(batch_y, num_classes=self.num_classes)
        return batch_x, batch_y

    def on_epoch_end(self):
        """"
        Called at the end of each epoch
        """
        # if required, shuffle your data after each epoch
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            # TODO shuffle data
            # you might find np.random.shuffle useful here
            np.random.shuffle(self.indices)


dg = Minc2500DataGenerator("D:/Files/Progs/Deep Learning/MaterialClassifier/data", 200, (256, 256, 3), 10, True)

batch_x, batch_y = dg[0]
test_x, test_y = dg.test_data()

fig, axes = plt.subplots(nrows=1, ncols=6, figsize=[16, 9])
for i in range(len(axes) // 2):
    axes[i].set_title(CLASSES[batch_y[i]])
    axes[i].imshow(batch_x[i])
for i in range(len(axes) // 2, len(axes)):
    axes[i].set_title("T:" + CLASSES[test_y[i]])
    axes[i].imshow(test_x[i])
plt.show()
print()