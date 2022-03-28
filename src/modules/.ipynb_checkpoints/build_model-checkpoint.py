from tensorflow.keras.layers import Bidirectional, Dense, Input, LSTM, Embedding
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

class buildModel():
    '''
    The model being used here is a modified U-Net. A U-Net consists of an encoder (downsampler) and decoder (upsampler). In-order to learn robust features and reduce the number of trainable parameters, you will use a pretrained model - EfficientNetV2B0 - as the encoder. For the decoder, you will use the upsample block, which is already implemented in the pix2pix example in the TensorFlow Examples repo. (Check out the pix2pix: Image-to-image translation with a conditional GAN tutorial in a notebook.)
    ----------

    Returns
    -------
    self.model:
        Deep learning based Model
    
    '''
    def __init__(self,train_data=None, modelName = 'EfficientNetV2B0', input_shape = [256, 256, 3], numOutClass = 59,pre_weight_flag = False, weights="imagenet", include_top=False, pooling=None, alpha=1.0,depth_multiplier=1, dropout=0.001):
        self.train_data = train_data
        self.model_name = modelName
        self.base_model = []
        self.layers = []
        self.layer_names = []
        
        self.input_shape = input_shape
        self.include_top = include_top
        
        self.base_model_outputs = []
        self.down_stack = []
        self.up_stack  = []
        self.numOutClass = numOutClass
        self.pre_weight_flag = pre_weight_flag
        
        self.weights = weights
        self.height = input_shape[0]
        self.width = input_shape[1]
        self.channels = input_shape[2]
        self.include_top = include_top
        self.pooling = pooling
        self.alpha = alpha
        self.depth_multiplier = depth_multiplier
        self.dropout = dropout
        
        
        
        ##self.base_model, self.layers, self.layer_names
        
    def defineBackbone(self):
        '''
        Define the model
        ----------
        
        Returns
        -------
        
        '''
        self.create_base_model()
        # define backbone model
        
#         if self.model_name == 'EfficientNetV2B0':
#             self.base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(input_shape=self.input_shape, include_top=self.include_top)
#             self.layer_names = [

#             #######EfficientNetV2B0
#             'block1a_project_activation',   # 64x64
#             'block2b_expand_activation',    # 32x32
#             'block4a_expand_activation',    # 16x16
#             'block6a_expand_activation',    # 8x8
#             'block6h_project_conv',         # 4x4
#             ]
#         else:
#             self.base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B3(input_shape=self.input_shape, include_top=self.include_top)
#             self.layer_names = [

#                 #     #######EfficientNetV2B3
#                 "block1b_project_activation",      # 64x64
#                 "block2c_expand_activation",       # 32x32
#                 "block4a_expand_activation",       # 16x16
#                 "block6a_expand_activation",       # 8x8
#                 "block6l_project_conv",            # 4x4
#                     ]
                
                
        # print(len(self.base_model.layers))

        # self.layers = [self.base_model.get_layer(name).output for name in self.layer_names]
        # self.layers = [self.base_model.get_layer(layer_name).output for layer_name in self.layer_names]

        # Create the feature extraction model
        self.down_stack = tf.keras.Model(inputs=self.base_model.input, outputs=self.layers)

        self.down_stack.trainable = False
        
        
        
    def defineUpStack(self):
        '''
        The decoder/upsampler is simply a series of upsample blocks implemented in TensorFlow examples.
        ----------
        
        Returns
        -------
        
        '''
        self.up_stack = [
            pix2pix.upsample(512, 3),  # 4x4 -> 8x8
            pix2pix.upsample(256, 3),  # 8x8 -> 16x16
            pix2pix.upsample(128, 3),  # 16x16 -> 32x32
            pix2pix.upsample(64, 3),   # 32x32 -> 64x64
        ]
        
    def unetModel(self):
          inputs = tf.keras.layers.Input(shape=self.input_shape)

          # Downsampling through the model
          skips = self.down_stack(inputs)
          x = skips[-1]
          skips = reversed(skips[:-1])

          # Upsampling and establishing the skip connections
          for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x =  tf.keras.layers.Concatenate()([x, skip])
            
          # This is the last layer of the model
          x = tf.keras.layers.Conv2DTranspose(filters=self.numOutClass, kernel_size=3, strides=2, padding='same')(x)           ## 64x64 -> 128x128
          self.model = tf.keras.Model(inputs=inputs, outputs=x)


    def defineModel(self):
        self.defineBackbone()
        self.defineUpStack()
        self.unetModel()
        
    def CompileModel(self):
        '''
        Compile the model
        ----------
        
        Returns
        -------
        
        '''
        
        self.model.compile(
        optimizer='adam',#tfa.optimizers.Yogi(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
#         return self.model
    
    def checkDataModel(self):
        '''
        Check for Model and Data compatibility
        '''
        
        example = next(iter(self.train_data))
        preds = self.model(example[0])
        plt.imshow(tf.keras.utils.array_to_img(example[0][60]))

        plt.colorbar()
        plt.show()
        pred_mask = tf.argmax(preds, axis=-1)
        pred_mask = tf.expand_dims(pred_mask, -1)
        plt.imshow(pred_mask[0])
        plt.colorbar()
        plt.show()
    
    def setupModel(self):
        '''
        Build the model
        ----------
        
        Returns
        -------
        
        '''
        if self.pre_weight_flag ==True:
            self.model = tf.keras.models.load_model('./models/SemImSeg_model_EfficientNetV2B0.h5')
        else:
                
            self.defineModel()
        self.CompileModel()
        self.model.summary()
        self.checkDataModel()
        return self.model
    

    

    ################################################################################
    # Backbone
    ################################################################################
    def create_base_model(self):
        # print (self.height,self.width,self.channels)
        if not isinstance(self.height, int) or not isinstance(self.width, int) or not isinstance(self.channels, int):
            raise TypeError("'height', 'width' and 'channels' need to be of type 'int'")

        if self.channels <= 0:
            raise ValueError(f"'channels' must be greater of equal to 1 but given was {self.channels}")

        self.input_shape = [self.height, self.width, self.channels]

        if self.model_name.lower() == "densenet121":
            if self.height <= 31 or self.width <= 31:
                raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
            self.base_model = tf.keras.applications.DenseNet121(include_top=self.include_top, weights=self.weights, input_shape=self.input_shape, pooling=self.pooling)
            self.layer_names = ["conv1/relu", "pool2_relu", "pool3_relu", "pool4_relu", "relu"]
            
            
        elif self.model_name.lower() == "densenet169":
            if self.height <= 31 or self.width <= 31:
                raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
            self.base_model = tf.keras.applications.DenseNet169(include_top=self.include_top, weights=self.weights, input_shape=self.input_shape, pooling=self.pooling)
            self.layer_names = ["conv1/relu", "pool2_relu", "pool3_relu", "pool4_relu", "relu"]
            
            
        elif self.model_name.lower() == "densenet201":
            if self.height <= 31 or self.width <= 31:
                raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
            self.base_model = tf.keras.applications.DenseNet201(include_top=self.include_top, weights=self.weights, input_shape=self.input_shape, pooling=self.pooling)
            self.layer_names = ["conv1/relu", "pool2_relu", "pool3_relu", "pool4_relu", "relu"]
            
            
        elif self.model_name.lower() == "efficientnetb0":
            if self.height <= 31 or self.width <= 31:
                raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
            self.base_model = tf.keras.applications.EfficientNetB0(include_top=self.include_top, weights=self.weights, input_shape=self.input_shape, pooling=self.pooling)
            self.layer_names = ["block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation", "top_activation"]
            
            
        elif self.model_name.lower() == "efficientnetb1":
            if self.height <= 31 or self.width <= 31:
                raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
            self.base_model = tf.keras.applications.EfficientNetB1(include_top=self.include_top, weights=self.weights, input_shape=self.input_shape, pooling=self.pooling)
            self.layer_names = ["block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation", "top_activation"]
            
            
        elif self.model_name.lower() == "efficientnetb2":
            if self.height <= 31 or self.width <= 31:
                raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
            self.base_model = tf.keras.applications.EfficientNetB2(include_top=self.include_top, weights=self.weights, input_shape=self.input_shape, pooling=self.pooling)
            self.layer_names = ["block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation", "top_activation"]
            
            
        elif self.model_name.lower() == "efficientnetb3":
            if self.height <= 31 or self.width <= 31:
                raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
            self.base_model = tf.keras.applications.EfficientNetB3(include_top=self.include_top, weights=self.weights, input_shape=self.input_shape, pooling=self.pooling)
            layer_names = ["block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation", "top_activation"]
            
            
        elif self.model_name.lower() == "efficientnetb4":
            if self.height <= 31 or self.width <= 31:
                raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
            self.base_model = tf.keras.applications.EfficientNetB4(include_top=self.include_top, weights=self.weights, input_shape=self.input_shape, pooling=self.pooling)
            self.layer_names = ["block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation", "top_activation"]
            
            
        elif self.model_name.lower() == "efficientnetb5":
            if self.height <= 31 or self.width <= 31:
                raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
            self.base_model = tf.keras.applications.EfficientNetB5(include_top=self.include_top, weights=self.weights, input_shape=self.input_shape, pooling=self.pooling)
            self.layer_names = ["block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation", "top_activation"]
            
            
        elif self.model_name.lower() == "efficientnetb6":
            if self.height <= 31 or self.width <= 31:
                raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
            self.base_model = tf.keras.applications.EfficientNetB6(include_top=self.include_top, weights=self.weights, input_shape=self.input_shape, pooling=self.pooling)
            self.layer_names = ["block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation", "top_activation"]
            
            
        elif self.model_name.lower() == "efficientnetb7":
            if self.height <= 31 or self.width <= 31:
                raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
            self.base_model = tf.keras.applications.EfficientNetB7(include_top=self.include_top, weights=self.weights, input_shape=self.input_shape, pooling=self.pooling)
            self.layer_names = ["block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation", "top_activation"]
            
 
        elif self.model_name.lower() == "efficientnetv2b0":
            if self.height <= 31 or self.width <= 31:
                raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
            self.base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=self.include_top, weights=self.weights, input_shape=self.input_shape, pooling=self.pooling)
            self.layer_names = ["block1a_project_activation", "block2b_expand_activation", "block4a_expand_activation", "block6a_expand_activation", "block6h_project_conv"]
            
            
        elif self.model_name.lower() == "efficientnetv2b3":
            if self.height <= 31 or self.width <= 31:
                raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
            self.base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B3(include_top=self.include_top, weights=self.weights, input_shape=self.input_shape, pooling=self.pooling)
            self.layer_names = ["block1a_project_activation", "block2c_expand_activation", "block4a_expand_activation", "block6a_expand_activation", "block6l_project_conv"]
            
        elif self.model_name.lower() == "mobilenet":
            if self.height <= 31 or self.width <= 31:
                raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
            self.base_model = tf.keras.applications.MobileNet(include_top=self.include_top, weights=self.weights, input_shape=self.input_shape, pooling=self.pooling, alpha=alpha, depth_multiplier=depth_multiplier, dropout=dropout)
            self.layer_names = ["conv_pw_1_relu", "conv_pw_3_relu", "conv_pw_5_relu", "conv_pw_11_relu", "conv_pw_13_relu"]
            
            
        elif self.model_name.lower() == "mobilenetv2":
            if self.height <= 31 or self.width <= 31:
                raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
            self.base_model = tf.keras.applications.MobileNetV2(include_top=self.include_top, weights=self.weights, input_shape=self.input_shape, pooling=self.pooling, alpha=alpha)
            self.layer_names = ["block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu", "block_13_expand_relu", "out_relu"]
            
            
        elif self.model_name.lower() == "nasnetlarge":
            if self.height <= 31 or self.width <= 31:
                raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
            self.base_model = tf.keras.applications.NASNetLarge(include_top=self.include_top, weights=self.weights, input_shape=self.input_shape, pooling=self.pooling)
            self.layer_names = ["zero_padding2d", "cropping2d_1", "cropping2d_2", "cropping2d_3", "activation_650"]
            
            
        elif self.model_name.lower() == "nasnetmobile":
            if self.height <= 31 or self.width <= 31:
                raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
            self.base_model = tf.keras.applications.NASNetMobile(include_top=self.include_top, weights=self.weights, input_shape=self.input_shape, pooling=self.pooling)
            self.layer_names =["zero_padding2d_4", "cropping2d_5", "cropping2d_6", "cropping2d_7", "activation_838"]
            
            
        elif self.model_name.lower() == "resnet50":
            if self.height <= 31 or self.width <= 31:
                raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
            self.base_model = tf.keras.applications.ResNet50(include_top=self.include_top, weights=self.weights, input_shape=self.input_shape, pooling=self.pooling)
            self.layer_names = ["conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
            
            
        elif self.model_name.lower() == "resnet50v2":
            if self.height <= 31 or self.width <= 31:
                raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
            self.base_model = tf.keras.applications.ResNet50V2(include_top=self.include_top, weights=self.weights, input_shape=self.input_shape, pooling=self.pooling)
            self.layer_names = ["conv1_conv", "conv2_block3_preact_relu", "conv3_block4_preact_relu", "conv4_block6_preact_relu", "post_relu"]
            
            
        elif self.model_name.lower() == "resnet101":
            if self.height <= 31 or self.width <= 31:
                raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
            self.base_model = tf.keras.applications.ResNet101(include_top=self.include_top, weights=self.weights, input_shape=self.input_shape, pooling=self.pooling)
            self.layer_names = ["conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block23_out", "conv5_block3_out"]
            
            
        elif self.model_name.lower() == "resnet101v2":
            if self.height <= 31 or self.width <= 31:
                raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
            self.base_model = tf.keras.applications.ResNet101V2(include_top=self.include_top, weights=self.weights, input_shape=self.input_shape, pooling=self.pooling)
            self.layer_names = ["conv1_conv", "conv2_block3_preact_relu", "conv3_block4_preact_relu", "conv4_block23_preact_relu", "post_relu"]
            
            
        elif self.model_name.lower() == "resnet152":
            if self.height <= 31 or self.width <= 31:
                raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
            self.base_model = tf.keras.applications.ResNet152(include_top=self.include_top, weights=self.weights, input_shape=self.input_shape, pooling=self.pooling)
            self.layer_names = ["conv1_relu", "conv2_block3_out", "conv3_block8_out", "conv4_block36_out", "conv5_block3_out"]
            
            
        elif self.model_name.lower() == "resnet152v2":
            if self.height <= 31 or self.width <= 31:
                raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
            self.base_model = tf.keras.applications.ResNet152V2(include_top=self.include_top, weights=self.weights, input_shape=self.input_shape, pooling=self.pooling)
            self.layer_names = ["conv1_conv", "conv2_block3_preact_relu", "conv3_block8_preact_relu", "conv4_block36_preact_relu", "post_relu"]
            
            
        elif self.model_name.lower() == "vgg16":
            if self.height <= 31 or self.width <= 31:
                raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
            self.base_model = tf.keras.applications.VGG16(include_top=self.include_top, weights=self.weights, input_shape=self.input_shape, pooling=self.pooling)
            self.layer_names = ["block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3", "block5_pool"]
        elif self.model_name.lower() == "vgg19":
            if self.height <= 31 or self.width <= 31:
                raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
            self.base_model = tf.keras.applications.VGG19(include_top=self.include_top, weights=self.weights, input_shape=self.input_shape, pooling=self.pooling)
            self.layer_names = ["block2_conv2", "block3_conv4", "block4_conv4", "block5_conv4", "block5_pool"]
            
            
        elif self.model_name.lower() == "xception":
            if height <= 70 or width <= 70:
                raise ValueError("Parameters 'height' and width' should not be smaller than 71.")
            self.base_model = tf.keras.applications.Xception(include_top=self.include_top, weights=self.weights, input_shape=self.input_shape, pooling=self.pooling)
            self.layer_names = ["block2_sepconv2_act", "block3_sepconv2_act", "block4_sepconv2_act", "block13_sepconv2_act", "block14_sepconv2_act"]
            
            
        else:
            raise ValueError("'name' should be one of 'densenet121', 'densenet169', 'densenet201', 'efficientnetb0', 'efficientnetb1', 'efficientnetb2', \
                    'efficientnetb3', 'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7', 'efficientnetv2b0','efficientnetv2b3','mobilenet', 'mobilenetv2', 'nasnetlarge', 'nasnetmobile', \
                    'resnet50', 'resnet50v2', 'resnet101', 'resnet101v2', 'resnet152', 'resnet152v2', 'vgg16', 'vgg19' or 'xception'.")

        self.layers = [self.base_model.get_layer(layer_name).output for layer_name in self.layer_names]

    
        

        
        

