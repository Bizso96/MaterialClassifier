import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from config import CLASSES
from datagenerator import Minc2500DataGenerator


class ModelTrainer():
    def __init__(self, db_dir, batch_size, input_shape):
        self._db_dir = db_dir
        self._batch_size = batch_size
        self._input_shape = input_shape

        self.minc2500_data_generator = Minc2500DataGenerator("D:/Files/Progs/Deep Learning/MaterialClassifier/data", batch_size, input_shape, len(CLASSES), True)

        self._model = None

    def start_train(self):
        pass

    def initilize_model(self):
        pass