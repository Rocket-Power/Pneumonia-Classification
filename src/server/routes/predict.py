import tensorflow as tf
from keras.utils import load_img, img_to_array
import numpy as np


def makePrediction(model):
  # read in image
  img = load_img('./image/image.jpeg', target_size = (256, 256))
  input = img_to_array(img)
  print(type(input))
  print(input.shape)
  input_arr = np.array([input])
  print(type(input_arr))
  print(input_arr.shape)
  # make prediction
  prediction = model.predict(input_arr)

  

  return prediction
