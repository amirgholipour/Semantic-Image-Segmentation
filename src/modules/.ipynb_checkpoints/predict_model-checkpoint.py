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
        print("Loading model",os.getpid())
        
        # print(os.getcwd())
        # self.cwd_path = str(Path(os.getcwd()).parents[1])
        self.cwd_path = os.getcwd()
        # print(self.cwd_path )
        self.model = tf.keras.models.load_model(self.cwd_path+self.model_path)
        self.loaded = True
        print("Loaded model")



    def predict(self):
        img, mask = next(iter(self.val))
        self.load()
        pred = self.model.predict(img)
        plt.figure(figsize=(10,5))
        for i in pred:

            print('####')
            plt.subplot(121)
            i = tf.argmax(i, axis=-1)
            plt.imshow(i,cmap='jet')
            plt.axis('off')
            plt.title('Prediction')
            break
        plt.subplot(122)
        plt.imshow(mask[0], cmap='jet')
        plt.axis('off')
        plt.title('Ground Truth')
        plt.show()