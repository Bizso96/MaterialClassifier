import os
import cv2
import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from config import CLASSES
from datagenerator import Minc2500DataGenerator


class ModelTrainer():
    def __init__(self, db_dir, batch_size, input_shape, train_count=2000, test_count=500):
        self._db_dir = db_dir
        self._batch_size = batch_size
        self._input_shape = input_shape
        self._number_of_classes = len(CLASSES)

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

        history = self._model.fit(x=self.minc2500_data_generator, epochs=20, validation_data=(test_images, test_labels))

        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.1, 1])
        plt.legend(loc='lower right')
        plt.show()

        test_loss, test_acc = self._model.evaluate(test_images, test_labels, verbose=2)
        print(test_acc)

        self._model.save(f'models/{self._model.name}')

    def initilize_model(self, model_name):
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


model_trainer = ModelTrainer("D:/Files/Progs/Deep Learning/MaterialClassifier/data", 250, (64, 64, 3), train_count=2000, test_count=500)
model_trainer.initilize_model('model1')
model_trainer.start_train()
