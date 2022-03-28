import sys
import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class predictor(object):
    
    def __init__(self, val_data = None, model_path='/models/SemImSeg_model_EfficientNetV2B0.h5'):
        self.model_path = model_path
        self.val = val_data
    def load(self):
        print("Loading the model...")
        
        # print(os.getcwd())
        # self.cwd_path = str(Path(os.getcwd()).parents[1])
        self.cwd_path = os.getcwd()
        # print(self.cwd_path )
        self.model = tf.keras.models.load_model(self.model_path)
        self.loaded = True
        print("The model has loaded!!!")



    def predict(self):
        img, mask = next(iter(self.val))
        self.load()
        pred = self.model.predict(img)
        plt.figure(figsize=(10,5))
        pred = tf.argmax(pred, axis=-1)
        pred = pred[..., tf.newaxis]
        plt.subplot(131)
        plt.imshow(tf.keras.utils.array_to_img(pred[0]), cmap='jet')
        plt.axis('off')
        plt.title('Predicted Mask')
        plt.subplot(132)
        plt.imshow(tf.keras.utils.array_to_img(mask[0]), cmap='jet')
        plt.axis('off')
        plt.title('Ground Truth')
        plt.subplot(133)
        plt.imshow(tf.keras.utils.array_to_img(img[0]))
        plt.axis('off')
        plt.title('Original Image')
        plt.show()