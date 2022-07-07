# loading libraries
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np


class Stage1:
    def detect_fruit(self, file_path):

        #load stage 1 model
        model = load_model('Models\FRU-MDL-S1v1.h5', compile=False)

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        file_path = file_path


        image = Image.open(file_path)
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)

        return prediction
    
    def get_prediction(self, prediction):
        grt_ind = prediction.argmax()

        if grt_ind == 0:
            op_msg = "Fruit Detected: Banana 🍌"
        elif grt_ind == 1:
            op_msg = "Fruit Detected: Mango 🥭"
        elif grt_ind == 2:
            op_msg = 'Fruit detected: Papaya '
        else:
            print('💡I Dont know wth is that')

        return grt_ind, op_msg

class Stage2:
    def detect_banana_ripeness(self, file_path):

        # load stage 2.1 model
        model = load_model('Models\FRU-MDL-S2.1v1.h5', compile=False)

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        file_path = file_path


        image = Image.open(file_path)
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)

        # detect ripeness level
        grt_ind = prediction.argmax()

        if grt_ind == 0:
            op_msg = "Ripeness: Overripe Banana"
        elif grt_ind == 1:
            op_msg = "Ripeness: Ripe Banana"
        elif grt_ind == 2:
            op_msg = "Ripeness: Unripe Banana"
        else:
            print('💡I Dont know wth is that')
        
        return grt_ind, op_msg
    
    def detect_mango_ripeness(self, file_path):

        # load stage 2.2 model
        model = load_model('Models\FRU-MDL-S2.2v1.h5', compile=False)

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        file_path = file_path


        image = Image.open(file_path)
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)

        # detect ripeness level
        grt_ind = prediction.argmax()

        if grt_ind == 0:
            op_msg = "Ripeness: Overripe Mango"
        elif grt_ind == 1:
            op_msg = "Ripeness: Ripe Mango"
        elif grt_ind == 2:
            op_msg = "Ripeness: Unripe Mango"
        else:
            print('💡I Dont know wth is that')
        
        return grt_ind, op_msg
    
    def detect_papaya_ripeness(self, file_path):

        # load stage 2.3 model
        model = load_model('Models\FRU-MDL-S2.3v1.h5', compile=False)

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        file_path = file_path


        image = Image.open(file_path)
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)

        # detect ripeness level
        grt_ind = prediction.argmax()

        if grt_ind == 0:
            op_msg = "Ripeness: Overripe Papaya"
        elif grt_ind == 1:
            op_msg = "Ripeness: Ripe Papaya"
        elif grt_ind == 2:
            op_msg = "Ripeness: Unripe Papaya"
        else:
            print('💡I Dont know wth is that')
        
        return grt_ind, op_msg