import os
import pickle

import cv2
import keras
import numpy as np
import tempfile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model

from config import *
from datagenerator import DataGenerator

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


class ModelTrainer:
    def __init__(self, train_data_generator, validation_data_generator, test_data_generator, batch_size, input_shape, epochs_count=20):
        self._batch_size = batch_size
        self._input_shape = input_shape
        self._number_of_classes = len(CLASSES)
        self._epochs_count = epochs_count

        self.train_data_generator = train_data_generator
        self.validation_data_generator = validation_data_generator
        self.test_data_generator = test_data_generator

        self._model = None

    def start_train(self):
        if self._model is None:
            print("Model not initialized")
            return

        self._model.compile(optimizer='adam',
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=['accuracy'])

        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        history = self._model.fit(
            x=self.train_data_generator,
            epochs=self._epochs_count,
            validation_data=self.test_data_generator,
            callbacks=[early_stop_callback])

        self.save_model_data(self._model, history)

        test_loss, test_acc = self._model.evaluate(x=self.test_data_generator, verbose=2)
        print(test_acc)


    def start_train_transfer(self):
        if self._model is None:
            print("Model not initialized")
            return

        self._model.compile(optimizer='adam',
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=['accuracy'])

        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        history = self._model.fit(
            x=self.train_data_generator,
            epochs=self._epochs_count,
            validation_data=self.validation_data_generator,
            callbacks=[early_stop_callback])

        with open(f'histories/{self._model.name}_pre.pickle', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        self.save_model_data(self._model, history, self._model.name + "_frozen")

        test_loss, test_acc = self._model.evaluate(self.test_data_generator, verbose=2)
        print(test_acc)

        self._model.trainable = True
        self._model.summary()

        self._model.compile(
            optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        history = self._model.fit(
            x=self.train_data_generator,
            epochs=self._epochs_count,
            validation_data=self.test_data_generator,
            callbacks=[early_stop_callback])

        self.save_model_data(self._model, history, self._model.name + "_fine_tuned")

    def save_model_data(self, model, history, name=None):
        if name is None:
            name = model.name
        model.save(f'models/{name}', save_format='h5')
        plot_model(model, f"images/{name}.jpg", show_shapes=True)

        with open(f'histories/{name}.pickle', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.1, 1])
        plt.legend(loc='lower right')

        plt.savefig(f'plots/{name}_plot.jpg')


    def initialize_model_resnet50_no_weights_base(self, model_name, image_augment=False):
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights=None,
            input_tensor=None,
            input_shape=self._input_shape
        )

        base_model.trainable = True

        inputs = keras.Input(shape=self._input_shape)
        if image_augment:
            x = tf.keras.layers.RandomFlip()(inputs)
            x = tf.keras.layers.RandomRotation(factor=(-0.2, 0.2))(x)
            x = base_model(x, training=True)
        else:
            x = base_model(inputs, training=True)

        x = layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(self._number_of_classes)(x)

        self._model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

        self._model.summary()

    def initialize_model_resnet50_no_weights_regularization(self, model_name, image_augment=False):
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights=None,
            input_tensor=None,
            input_shape=self._input_shape
        )

        base_model.trainable = True

        alpha = 1e-5

        regularizer = tf.keras.regularizers.l2(alpha)

        for layer in base_model.layers:
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    setattr(layer, attr, regularizer)

        model_json = base_model.to_json()

        # Save the weights before reloading the model.
        tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
        base_model.save_weights(tmp_weights_path)

        # load the model from the config
        base_model = tf.keras.models.model_from_json(model_json)

        # Reload the model weights
        base_model.load_weights(tmp_weights_path, by_name=True)

        inputs = keras.Input(shape=self._input_shape)
        if image_augment:
            x = tf.keras.layers.RandomFlip()(inputs)
            x = tf.keras.layers.RandomRotation(factor=(-0.2, 0.2))(x)
            x = base_model(x, training=True)
        else:
            x = base_model(inputs, training=True)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.1)(x)
        outputs = keras.layers.Dense(self._number_of_classes)(x)

        self._model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

        self._model.summary()

    def initialize_model_resnet50_weights_base(self, model_name, image_augment=False):
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=self._input_shape
        )

        base_model.trainable = False

        inputs = keras.Input(shape=self._input_shape)
        if image_augment:
            x = tf.keras.layers.RandomFlip()(inputs)
            x = tf.keras.layers.RandomRotation(factor=(-0.2, 0.2))(x)
            x = base_model(x, training=False)
        else:
            x = base_model(inputs, training=False)

        x = layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(self._number_of_classes)(x)

        self._model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

        self._model.summary()

    def initialize_model_inceptionV3_no_weights_base(self, model_name, image_augment=False):
        base_model = tf.keras.applications.InceptionV3(
            include_top=False,
            weights=None,
            input_tensor=None,
            input_shape=self._input_shape
        )

        base_model.trainable = True

        inputs = keras.Input(shape=self._input_shape)
        if image_augment:
            x = tf.keras.layers.RandomFlip()(inputs)
            x = tf.keras.layers.RandomRotation(factor=(-0.2, 0.2))(x)
            x = base_model(x, training=True)
        else:
            x = base_model(inputs, training=True)

        x = layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(self._number_of_classes)(x)

        self._model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

        self._model.summary()

    def initialize_model_inceptionV3_no_weights_regularization(self, model_name, image_augment=False):
        base_model = tf.keras.applications.InceptionV3(
            include_top=False,
            weights=None,
            input_tensor=None,
            input_shape=self._input_shape
        )

        base_model.trainable = True

        alpha = 1e-5

        regularizer = tf.keras.regularizers.l2(alpha)

        for layer in base_model.layers:
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    setattr(layer, attr, regularizer)

        model_json = base_model.to_json()

        # Save the weights before reloading the model.
        tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
        base_model.save_weights(tmp_weights_path)

        # load the model from the config
        base_model = tf.keras.models.model_from_json(model_json)

        # Reload the model weights
        base_model.load_weights(tmp_weights_path, by_name=True)

        inputs = keras.Input(shape=self._input_shape)
        if image_augment:
            x = tf.keras.layers.RandomFlip()(inputs)
            x = tf.keras.layers.RandomRotation(factor=(-0.2, 0.2))(x)
            x = base_model(x, training=True)
        else:
            x = base_model(inputs, training=True)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.1)(x)
        outputs = keras.layers.Dense(self._number_of_classes)(x)

        self._model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

        self._model.summary()

    def initialize_model_inceptionV3_weights_base(self, model_name, image_augment=False):
        base_model = tf.keras.applications.InceptionV3(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=self._input_shape
        )

        base_model.trainable = False

        inputs = keras.Input(shape=self._input_shape)
        if image_augment:
            x = tf.keras.layers.RandomFlip()(inputs)
            x = tf.keras.layers.RandomRotation(factor=(-0.2, 0.2))(x)
            x = base_model(x, training=False)
        else:
            x = base_model(inputs, training=False)

        x = layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(self._number_of_classes)(x)

        self._model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

        self._model.summary()


def minc2500_train():
    start = 0
    train_data_end = MINC2500_TRAIN_AMOUNT
    validation_data_end = train_data_end + MINC2500_VALIDATION_AMOUNT
    test_data_end = validation_data_end + MINC2500_TEST_AMOUNT

    root_dir = f'{DATA_ROOT_PATH}/{MINC2500_PATH}'

    minc2500_train_data_generator = DataGenerator(root_dir, BATCH_SIZE, INPUT_SHAPE, len(CLASSES), 0, train_data_end)
    minc2500_validation_data_generator = DataGenerator(root_dir, BATCH_SIZE, INPUT_SHAPE, len(CLASSES), train_data_end,
                                                       validation_data_end)
    minc2500_test_data_generator = DataGenerator(root_dir, BATCH_SIZE, INPUT_SHAPE, len(CLASSES), validation_data_end,
                                                 test_data_end)

    model_trainer = ModelTrainer(
        minc2500_train_data_generator,
        minc2500_validation_data_generator,
        minc2500_test_data_generator,
        batch_size=BATCH_SIZE,
        input_shape=INPUT_SHAPE,
        epochs_count=EPOCHS_COUNT
    )

    return model_trainer

def fmd_train():
    start = 0
    train_data_end = FMD_TRAIN_AMOUNT
    validation_data_end = train_data_end + FMD_VALIDATION_AMOUNT
    test_data_end = validation_data_end + FMD_TEST_AMOUNT

    root_dir = f'{DATA_ROOT_PATH}/{FMD_PATH}'

    fmd_train_data_generator = DataGenerator(root_dir, BATCH_SIZE, INPUT_SHAPE, len(CLASSES), 0,
                                                  train_data_end)
    fmd_validation_data_generator = DataGenerator(root_dir, BATCH_SIZE, INPUT_SHAPE,
                                                       len(CLASSES), train_data_end,
                                                       validation_data_end)
    fmd_test_data_generator = DataGenerator(root_dir, BATCH_SIZE, INPUT_SHAPE, len(CLASSES),
                                                 validation_data_end,
                                                 test_data_end)

    model_trainer = ModelTrainer(
        fmd_train_data_generator,
        fmd_validation_data_generator,
        fmd_test_data_generator,
        batch_size=BATCH_SIZE,
        input_shape=INPUT_SHAPE,
        epochs_count=EPOCHS_COUNT
    )

    return model_trainer


with tf.device('/GPU:0'):
    model_trainer = minc2500_train()
    model_trainer.initialize_model_inceptionV3_weights_base('model_minc2500_inceptionV3_weights_base', image_augment=False)
    model_trainer.start_train_transfer()
