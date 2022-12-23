import tensorflow as tf
from keras.utils import load_img, img_to_array
import numpy as np


def makePrediction(model):
  try:
    # read in image to model target size and set to ndarray 
    img = load_img('./image/image.jpeg', target_size = (256, 256))
    input = img_to_array(img)
    input_arr = np.array([input])

    # make prediction
    prediction = model.predict(input_arr)
    # return predicted output
    return prediction

  except Exception as e:
    print(f'Error loading / rendering image in predict.py: {e}')
    return None