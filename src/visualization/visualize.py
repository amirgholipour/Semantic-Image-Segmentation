import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output


class visualizeData():
    '''
    Turn raw data into features for modeling
    ----------

    Returns
    -------
    self.final_set:
        Features for modeling purpose
    self.labels:
        Output labels of the features
    enc: 
        Ordinal Encoder definition file
    ohe:
        One hot  Encoder definition file
    '''
    def __init__(self, normalize= False, display_list = None, pred_mask = None,dataset = None, model= None,sample_image = None,sample_mask = None):
        self.normalize= normalize
        self.display_list = display_list
        self.pred_mask = pred_mask
        self.dataset = dataset
        self.model = model
        self.sample_image = sample_image
        self.sample_mask = sample_mask
        self.title = ['Input Image', 'True Mask', 'Predicted Mask']
        
        
#         self.final_set,self.labels = self.build_data()
    def display(self):
            plt.figure(figsize=(15, 15))
            
            for i in range(len(self.display_list)):
                plt.subplot(1, len(self.display_list), i+1)
                plt.title(self.title[i])
                if float(tf.__version__[:3]) <=5:
                    plt.imshow(tf.keras.preprocessing.image.array_to_img(self.display_list[i]), cmap='jet') ## tensorflow 2.4
                
                else:
                    
                    plt.imshow(tf.keras.utils.array_to_img(self.display_list[i]), cmap='jet') ## tensorflow 2.8
                plt.axis('off')
            plt.show()
    def create_mask(self):
        self.pred_mask = tf.argmax(self.pred_mask, axis=-1)
        self.pred_mask = self.pred_mask[..., tf.newaxis]
        return self.pred_mask[0]
        
    def show_predictions(self, num=1):
        if self.dataset:
            for image, mask in self.dataset.take(num):
                self.pred_mask = self.model.predict(image)
                self.display_list = [image[0], mask[0], self.pred_mask]
                self.display()
        else:
            self.pred_mask = self.model.predict(self.sample_image[tf.newaxis, ...]) 
            self.display_list = [self.sample_image, self.sample_mask, self.create_mask()]
            self.display()
            
class DisplayCallback(tf.keras.callbacks.Callback):
      '''
      The callback defined below is used to observe how the model improves while it is training.

      '''
        
      def __init__(self, model = None, sample_image = None, sample_mask = None):
            self.model = model
            self.sample_image  = sample_image
            self.sample_mask =  sample_mask
      def on_epoch_end(self,epoch, logs=None):
        clear_output(wait=True)
        visualizeData(model = self.model,sample_image = self.sample_image, sample_mask = self.sample_mask).show_predictions()
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))