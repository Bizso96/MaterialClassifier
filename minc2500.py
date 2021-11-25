import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from config import *


class Minc2500:
    def __init__(self):
        self.__data_location = None
        self.__materials = [
            "fabric",
            "foliage",
            "glass",
            "leather",
            "metal",
            "paper",
            "plastic",
            "stone",
            "water",
            "wood"
        ]

        self.__train_data = {}
        self.__test_data = {}

    @property
    def data_location(self):
        return self.__data_location

    @data_location.setter
    def data_location(self, value):
        self.__data_location = value

    def read_data(self):
        if self.__data_location is None:
            raise Exception("Data location not set")

        i = 0

        for material in self.__materials:
            path = f'{self.__data_location}/images/{material}'

            print("Reading in: " + path)

            images = []
            i = 0

            for filename in os.listdir(path):
                if i >= min(IMAGES_PER_CATEGORY, 2500):
                    i = 0
                    break

                i += 1

                if filename.endswith(".jpg"):
                    # print(" " + filename)
                    img = cv2.imread(path + "/" + filename)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))

                    images.append(img)

            images = np.array(images)
            np.random.shuffle(images)

            split_index = int(images.shape[0] * TRAIN_TEST_RATIO)

            self.__train_data[material], self.__test_data[material] = np.split(images, [split_index])