# loading libraries
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np
import os

from misc import Misc
from essentials import Stage1, Stage2
from decay_detect import Stage3

# Just ML code not to be used by the webapp

# loading the model
model = keras.models.load_model('Models\FRU-MDL-S1v1.h5', compile=False)

# setting the tensorflow log messages to warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# initializing the objects
misc = Misc()
stage1 = Stage1()
stage2 = Stage2()
stage3 = Stage3()

# get the file and file path
file_path = misc.open_file()

# get the stage 1 prediction
prediction = stage1.detect_fruit(file_path)

# print the stage 1 prediction
stage1.get_prediction(prediction)

# call the banana ripeness detection model function from stage 2
if prediction.argmax() == 0:
    banana_ripeness = stage2.detect_banana_ripeness(file_path)
    stage3.detect_banana_decay(file_path, None)
elif prediction.argmax() == 1:
    mango_ripeness = stage2.detect_mango_ripeness(file_path)
    stage3.detect_mango_decay(file_path, mango_ripeness)
elif prediction.argmax() == 2:
    papaya_ripeness = stage2.detect_papaya_ripeness(file_path)
    stage3.detect_papaya_decay(file_path, papaya_ripeness)
else:
    print('💡I Dont know wth is that')
