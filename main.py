from keras import models
from matplotlib import pyplot as plt

from minc2500 import Minc2500
from config import *
# from datagenerator import Minc2500DataGenerator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf

import tensorflow as tf
import os
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))