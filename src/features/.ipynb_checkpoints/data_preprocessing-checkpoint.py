


import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


import tensorflow as tf

from sklearn.model_selection import train_test_split
from ..visualization.visualize import visualizeData, DisplayCallback


class preprocessData():
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
    def __init__(self, images = None,masks = None,normalize= False):
        self.normalize = normalize
        self.images = images
        self.masks = masks
        self.batch = 64
        self.at = tf.data.AUTOTUNE
        self.buffer  = 1000
        self.size = (256,256)
        

        
#         self.final_set,self.labels = self.build_data()
    def resizeImage(self,image):
        image = tf.cast(image, tf.float32)
        if self.normalize  == True:
            image = image/255.0
        # resize image
        image = tf.image.resize(image, self.size)
        return image
    #### Functions for augmentation
    def resizeMask(self, mask):
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.image.resize(mask, self.size)
        mask = tf.cast(mask, tf.uint8)
        return mask  
    
    def brightness(self,img, mask):
        img = tf.image.adjust_brightness(img, 0.1)
        return img, mask

    def gamma(self,img, mask):
        img = tf.image.adjust_gamma(img, 0.1)
        return img, mask

    def hue(self,img, mask):
        img = tf.image.adjust_hue(img, -0.1)
        return img, mask

    def crop(self,img, mask):
        img = tf.image.central_crop(img, 0.7)
        img = tf.image.resize(img, self.size)
        mask = tf.image.central_crop(mask, 0.7)
        mask = tf.image.resize(mask, self.size)
        mask = tf.cast(mask, tf.uint8)
        return img, mask

    def flip_hori(self,img, mask):
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
        return img, mask

    def flip_vert(self,img, mask):
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)
        return img, mask

    def rotate(self,img, mask):
        img = tf.image.rot90(img)
        mask = tf.image.rot90(mask)
        return img, mask
    
    def dataAugmentation (self):
        self.train = tf.data.Dataset.zip((self.train_X, self.train_y))
        self.val = tf.data.Dataset.zip((self.val_X, self.val_y))

        # perform augmentation on train data only

        a = self.train.map(self.brightness)
        b = self.train.map(self.gamma)
        c = self.train.map(self.hue)
        d = self.train.map(self.crop)
        e = self.train.map(self.flip_hori)
        f = self.train.map(self.flip_vert)
        g = self.train.map(self.rotate)

        self.train = self.train.concatenate(a)
        self.train = self.train.concatenate(b)
        self.train = self.train.concatenate(c)
        self.train = self.train.concatenate(d)
        self.train = self.train.concatenate(e)
        self.train = self.train.concatenate(f)
        self.train = self.train.concatenate(g)
    def splitTrainTest(self):
        '''
        Split Data for training and validation
        
        '''
        
        self.train_X, self.val_X,self.train_y, self.val_y = train_test_split(self.X,self.y, 
                                                              test_size=0.2, 
                                                              random_state=0
                                                             )
        self.train_X = tf.data.Dataset.from_tensor_slices(self.train_X)
        self.val_X = tf.data.Dataset.from_tensor_slices(self.val_X)

        self.train_y = tf.data.Dataset.from_tensor_slices(self.train_y)
        self.val_y = tf.data.Dataset.from_tensor_slices(self.val_y)

        print(self.train_X.element_spec, self.train_y.element_spec, self.val_X.element_spec, self.val_y.element_spec)
        
    def plot(self):
        for i in range(10,13):
              self.sample_image, self.sample_mask = self.X[i],self.y[i]

              # tf.keras.utils.save_img(
              #       'test_sample_'+str(i)+'.jpg', sample_image
              #   )
              # tf.keras.utils.save_img(
              #       'test_sample__mask_'+str(i)+'.jpg', sample_mask
              #   )  
              visualizeData(display_list = [self.sample_image, self.sample_mask]).display()
    def dataPreProcessing(self):
        self.X = [self.resizeImage(i) for i in self.images]
        self.y = [self.resizeMask(m) for m in self.masks]
        self.plot()
        
    
    
        print(len(self.X), len(self.y))
        self.splitTrainTest()
        self.dataAugmentation()
        ### Prepare data in the form of batch processing
        self.train = self.train.cache().shuffle(self.buffer).batch(self.batch).repeat()
        self.train = self.train.prefetch(buffer_size=self.at)
        self.val = self.val.batch(self.batch)
        return self.train, self.val,self.sample_image, self.sample_mask
        
        
    
    
    