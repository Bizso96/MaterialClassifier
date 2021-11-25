from minc2500 import Minc2500
from config import *

minc2500 = Minc2500()

minc2500.data_location = MINC2500_PATH
minc2500.read_data()

print("Done")
