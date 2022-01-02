import os
import pickle

import cv2
import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model

from config import CLASSES
from datagenerator import Minc2500DataGenerator

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


class ModelTrainer:
    def __init__(self, db_dir, batch_size, input_shape, train_count=2000, test_count=500, epochs_count=20):
        self._db_dir = db_dir
        self._batch_size = batch_size
        self._input_shape = input_shape
        self._number_of_classes = len(CLASSES)
        self._epochs_count = epochs_count

        self.minc2500_data_generator = Minc2500DataGenerator(db_dir, batch_size, input_shape, len(CLASSES), True, img_limit=train_count, test_count=test_count)

        self._model = None

    def start_train(self):
        if self._model is None:
            print("Model not initialized")
            return

        self._model.compile(optimizer='adam',
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=['accuracy'])

        test_images, test_labels = self.minc2500_data_generator.test_data()

        history = self._model.fit(x=self.minc2500_data_generator, epochs=self._epochs_count, validation_data=(test_images, test_labels))

        self.save_model_data(self._model, history)

        test_loss, test_acc = self._model.evaluate(test_images, test_labels, verbose=2)
        print(test_acc)

    def start_train_lr_schedule(self):
        if self._model is None:
            print("Model not initialized")
            return

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2,
            decay_steps=10000,
            decay_rate=0.9)
        optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)

        self._model.compile(optimizer=optimizer,
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=['accuracy'])

        test_images, test_labels = self.minc2500_data_generator.test_data()

        history = self._model.fit(x=self.minc2500_data_generator, epochs=self._epochs_count, validation_data=(test_images, test_labels))

        self.save_model_data(self._model, history)

        test_loss, test_acc = self._model.evaluate(test_images, test_labels, verbose=2)
        print(test_acc)

    def save_model_data(self, model, history):
        model.save(f'models/{model.name}', save_format='h5')
        plot_model(model, f"images/{model.name}.jpg", show_shapes=True)

        with open(f'histories/{model.name}.pickle', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.1, 1])
        plt.legend(loc='lower right')

        plt.savefig(f'plots/{self._model.name}_plot.jpg')

    def initialize_model1(self, model_name):
        inputs = keras.Input(shape=self._input_shape)
        x = layers.Conv2D(64, (3, 3), activation="relu")(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(self._number_of_classes)(x)

        self._model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

        self._model.summary()

    def initialize_model3(self, model_name):
        inputs = keras.Input(shape=self._input_shape)
        x = tf.keras.layers.RandomFlip()(inputs)
        x = layers.Conv2D(64, (3, 3), activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(self._number_of_classes)(x)

        self._model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

        self._model.summary()

    def initialize_model4(self, model_name):
        inputs = keras.Input(shape=self._input_shape)
        x = tf.keras.layers.RandomFlip()(inputs)
        x = layers.ZeroPadding2D(padding=(2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), kernel_regularizer=keras.regularizers.l2(0.001), activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(self._number_of_classes)(x)

        self._model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

        self._model.summary()

    def initialize_model5(self, model_name):
        inputs = keras.Input(shape=self._input_shape)
        x = tf.keras.layers.RandomRotation((-0.3, 0.3))(inputs)
        x = layers.ZeroPadding2D(padding=(2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), kernel_regularizer=keras.regularizers.l2(0.01), activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(self._number_of_classes)(x)

        self._model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

        self._model.summary()

    def initialize_model_resnet50(self, model_name):
        self._model = ResNet50()

        self._model.summary()


with tf.device('/GPU:0'):
    model_trainer = ModelTrainer("C:/Files", 125, (64, 64, 3), train_count=2000, test_count=500, epochs_count=20)
    model_trainer.initialize_model5('model7')
    model_trainer.start_train_lr_schedule()
