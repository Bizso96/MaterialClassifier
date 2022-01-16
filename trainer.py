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

        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

        history = self._model.fit(
            x=self.train_data_generator,
            epochs=self._epochs_count,
            validation_data=self.test_data_generator,
            callbacks=[early_stop_callback])

        self.save_model_data(self._model, history)

        test_loss, test_acc = self._model.evaluate(x=self.test_data_generator, verbose=2)
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



        history = self._model.fit(
            x=self.train_data_generator,
            epochs=self._epochs_count,
            validation_data=self.validation_data_generator)

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

        history = self._model.fit(x=self.train_data_generator, epochs=self._epochs_count, validation_data=self.validation_data_generator)

        with open(f'histories/{self._model.name}_pre.pickle', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        self._model.trainable = False

        self._model.compile(
            optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

        history = self._model.fit(
            x=self.train_data_generator,
            epochs=self._epochs_count,
            validation_data=self.test_data_generator,
            callbacks=[early_stop_callback])

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

    '''
    The chosen model is resnet50 is a 50 layers deep neural network meant to be used for image classification, 
    object localisation and object detection. The reason why I chose this NN was because I wanted to start replicating
    the study of Anca Sticlaru. She trained 8 different neural networks on different datasets, one of these being
    resnet50. For now I wanted to avoid the networks with very large numbers of parameters (e.g., resnet101, vgg16),
    and using resnet50 was more convenient because it appears in the tensorflow applications (unlike some of the other 
    networks used by her, like GoogleNet and VGG_CNN_S).
    
    The trained network's accuracy is much lower (~30% with finetuning) than the ones achieved by the simpler 
    architectures. It is quite likely however that the network is underfitted, and a higher accuracy can be achieved by 
    a longer training period. I kept the 20 epoch training period for consistency among the trains between the models
    '''
    def initialize_model_resnet50_no_weights_base(self, model_name):
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights=None,
            input_tensor=None,
            input_shape=self._input_shape
        )

        base_model.trainable = True

        inputs = keras.Input(shape=self._input_shape)

        x = base_model(inputs, training=True)
        x = layers.Flatten()(x)
        outputs = keras.layers.Dense(self._number_of_classes)(x)

        self._model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

        self._model.summary()

    def initialize_model_resnet50_no_weights_regularization(self, model_name):
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

        x = base_model(inputs, training=True)
        x = layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(self._number_of_classes)(x)

        self._model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

        self._model.summary()

    def initialize_model_resnet50_weights_base(self, model_name):
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=self._input_shape
        )

        base_model.trainable = False

        inputs = keras.Input(shape=self._input_shape)

        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(self._number_of_classes)(x)

        self._model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

        self._model.summary()


    def initialize_model_fmd(self, model_name):
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights=None,
            input_tensor=None,
            input_shape=self._input_shape
        )

        base_model.trainable = False

        inputs = keras.Input(shape=self._input_shape)
        x = tf.keras.layers.RandomFlip()(inputs)

        x = base_model(x, training=False)
        x = layers.Conv2D(64, (2, 2), activation="relu")(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu")(x)
        outputs = keras.layers.Dense(self._number_of_classes)(x)

        self._model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

        self._model.summary()


def minc2500_train():
    start = 0
    train_data_end = MINC2500_TRAIN_AMOUNT
    validation_data_end = train_data_end + MINC2500_VALIDATION_AMOUNT
    test_data_end = validation_data_end + MINC2500_TEST_AMOUNT

    root_dir = f'{DATA_ROOT_PATH}/{MINC2500_PATH}'

    minc2500_train_data_generator = DataGenerator(root_dir, MINC2500_BATCH_SIZE, MINC2500_INPUT_SHAPE, len(CLASSES), 0, train_data_end)
    minc2500_validation_data_generator = DataGenerator(root_dir, MINC2500_BATCH_SIZE, MINC2500_INPUT_SHAPE, len(CLASSES), train_data_end,
                                                       validation_data_end)
    minc2500_test_data_generator = DataGenerator(root_dir, MINC2500_BATCH_SIZE, MINC2500_INPUT_SHAPE, len(CLASSES), validation_data_end,
                                                 test_data_end)

    model_trainer = ModelTrainer(
        minc2500_train_data_generator,
        minc2500_validation_data_generator,
        minc2500_test_data_generator,
        batch_size=MINC2500_BATCH_SIZE,
        input_shape=MINC2500_INPUT_SHAPE,
        epochs_count=EPOCHS_COUNT
    )

    return model_trainer


with tf.device('/GPU:0'):
    model_trainer = minc2500_train()
    model_trainer.initialize_model_resnet50_no_weights_regularization('model_resnet50_no_weights_regularization')
    model_trainer.start_train()
