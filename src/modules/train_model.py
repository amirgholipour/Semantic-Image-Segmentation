import os
import tensorflow as tf
import subprocess
import joblib
import matplotlib.pyplot as plt
    

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    

class trainModel():
    '''
    Build Lstm model for tensorflow
    ----------

    Returns
    -------
    self.model:
        Deep learning based Model
    
    '''
    
    def __init__(self, model,train_data = None, test_data = None,validation_steps =50, step_per_epoch = 50 ,  val_subsplits = 5,  batch_size=64,epochs=70,sample_image = None,sample_mask = None,display_callback = None, fineTune=False,modelDir = None):
        self.model_checkpoint_callback = []
        self.modelDir = modelDir
        self.fineTune = fineTune
        if self.fineTune == True:
             self.model = tf.keras.models.load_model(self.modelDir)
             self.model.compile(
                    optimizer='adam',#tfa.optimizers.Yogi(learning_rate=0.001),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        else:
            
            self.model = model
        self.train_data = train_data

        self.test_data  = test_data

        self.batch_size = batch_size
        self.epochs = epochs
        self.val_subsplits = val_subsplits
        self.sample_image = sample_image
        self.sample_mask = sample_mask
        self.step_per_epoch = step_per_epoch
        self.validation_steps = validation_steps
        self.ckpp = self.modelDir
        self.display_callback = display_callback
        self.fineTune = fineTune
        self.modelDir = modelDir
        self.history = []
        

        
    def plotHistory(self):
        '''
            Plot Performance Curves
        '''
        history = self.history.history
        acc=history['accuracy']
        val_acc = history['val_accuracy']

        plt.plot(acc, '-', label='Training Accuracy')
        plt.plot(val_acc, '--', label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
        
    def modelTraining(self):
        '''
        Define the model
        ----------
        
        Returns
        -------
        
        '''


        # VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS
        ## Saving the best model
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.ckpp,
            #save_weights_only=True,
            # monitor='iou_score',
            monitor='val_accuracy',
            # monitor=['val_sparse_categorical_crossentropy'],
            mode='max',
            save_best_only=True)
        ## Early Stoping
        self.early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            # monitor='iou_score', 
            monitor='val_loss',
            patience=20, #monitor='val_loss', patience=20, 
        )
        ## Train the model and save the historical information
        self.history = self.model.fit(
            self.train_data,
            epochs=self.epochs,
              steps_per_epoch=self.step_per_epoch,
              validation_steps=self.validation_steps,
              validation_data=self.test_data,
              callbacks=[self.display_callback(self.model,self.sample_image,self.sample_mask),self.model_checkpoint_callback,self.early_stopping_callback])
        self.plotHistory()
        return self.model
    