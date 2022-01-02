import numpy
from keras import models
from matplotlib import pyplot as plt

from datagenerator import Minc2500DataGenerator
from minc2500 import Minc2500
from config import *
# from datagenerator import Minc2500DataGenerator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def ensemble(model1name, model2name, model3name):
    model1 = models.load_model(f"models/{model1name}")
    model2 = models.load_model(f"models/{model2name}")
    model3 = models.load_model(f"models/{model3name}")
    dg = Minc2500DataGenerator("D:/Files/Progs/Deep Learning/MaterialClassifier/data", 500, (64, 64, 3), 10, True)

    batch_x, batch_y = dg[0]

    fig, axes = plt.subplots(nrows=1, ncols=10, figsize=[16, 9])
    for i in range(len(axes)):
        axes[i].set_title(CLASSES[batch_y[i]])
        axes[i].imshow(batch_x[i])
    plt.show()

    numpy.set_printoptions(suppress=True)
    prediction1 = model1.predict(batch_x)
    prediction2 = model2.predict(batch_x)
    prediction3 = model3.predict(batch_x)

    prediction_avg = np.average(np.array([prediction1, prediction2, prediction3]), axis=0)

    final_predictions = np.argmax(prediction_avg, axis=1)

    matches = 0
    for i in range(len(final_predictions)):
        if final_predictions[i] == batch_y[i]:
            matches += 1

    # print(final_predictions)
    # for i in final_predictions:
    #     print(CLASSES[i])

    print("Accuracy: ", matches / len(final_predictions))



ensemble("model3", "model4", "model5")

