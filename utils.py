import tensorflow as tf
import numpy as np
import cv2


def read_image(path):
  x = cv2.imread(path, cv2.IMREAD_COLOR)
  y = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
  y = cv2.resize(y, (256,256))
  y = y/255.0
  x = cv2.resize(x, (256,256))
  x = x/255.0    #(256, 256, 3)
  return x, y

def read_mask(path):
  x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  x = cv2.resize(x, (256,256))
  x = np.expand_dims(x, axis = -1)    #(256, 256, 1)
  return x

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask



