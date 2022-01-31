import tensorflow as tf
# from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Concatenate, Input, Flatten, Dense
# from tensorflow.keras.layers import Conv2DTranspose, Concatenate, UpSampling2D, LeakyReLU
# from tensorflow.keras.layers import Reshape, Lambda
# from tensorflow.keras.models import Model
# import tensorflow_addons as tfa
# from tensorflow_addons.layers import SpectralNormalization
# from tensorflow.keras.layers.experimental.preprocessing import Resizing

from tensorflow.keras.layers import Input, Add, Dense, Flatten, Activation, LeakyReLU, ReLU, Reshape
from tensorflow.keras.layers import Conv2D, BatchNormalization, GlobalAveragePooling2D, MaxPooling2D,Conv2DTranspose
from tensorflow.keras.models import Model
from loss import *


class Encoder(tf.keras.Model):

    def __init__(self, img_shape=(256,256,1), layer_name="", GAP=False):
        super().__init__(name=layer_name)
        self.img_shape = img_shape
        self.channel = img_shape[-1]
        self.layer_name = layer_name
        self.GAP = GAP
    
    def build(self):
        
        input_layer = Input(name=f'{self.layer_name}_input', shape=self.img_shape)
        x = input_layer
        
        n_filters = 32
        for i in range(1,5+1):

            if i==1:
                x = Conv2D(n_filters*(2**i), (5,5), strides=(1,1), padding='same', name=f'{self.layer_name}_conv_{i}', kernel_regularizer='l2')(x)
            else:
                x = Conv2D(n_filters*(2**i), (3,3), strides=(2,2), padding='same', name=f'{self.layer_name}_conv_{i}', kernel_regularizer='l2')(x)
                x = BatchNormalization(name=f'{self.layer_name}_norm_{i}')(x)
            
            x = LeakyReLU(name=f'{self.layer_name}_leaky_{i}')(x)
        
        if self.GAP:
            x = GlobalAveragePooling2D(name=f'{self.layer_name}_output')(x)
        
        return tf.keras.Model(inputs=input_layer, outputs=x)


class Decoder(tf.keras.Model):

    def __init__(self, z_shape, img_shape=(512,512,1)):
        super().__init__()
        self.z_shape = z_shape
        self.img_shape = img_shape
        self.channel = img_shape[-1]
    
    def build(self):
        
        height = self.img_shape[0]
        width = self.img_shape[1]
                
        z_input = Input(name='g_decoder_input', shape=self.z_shape)
        x = z_input

        y = Dense(height*width*2, name='dense')(x) # 2 = 128 / 8 / 8
        y = Reshape((height//8, width//8, 128), name='de_reshape')(y)
        
        n_filters = 128
        for i in range(1,3+1):
            
            y = Conv2DTranspose(n_filters//(2**i), (3,3), strides=(2,2), padding='same', name=f'deconv_{i}', kernel_regularizer='l2')(y)
            y = ReLU(name=f'gd_{i}')(y)
          
        # tanhになってたのをsigmoidに修正
        # y = Conv2DTranspose(self.channel, (1,1), strides=(1,1), padding='same', name='decoder_deconv_output', kernel_regularizer='l2', activation='sigmoid')(y)
        img = Conv2DTranspose(self.channel, (1,1), strides=(1,1), padding='same', name='decoder_deconv_output', kernel_regularizer='l2', activation='sigmoid')(y)
        
        return tf.keras.Model(inputs=z_input, outputs=img)

# z_shape = g_encoder.output.shape[1:]
# decoder = G_Decoder(z_shape=z_shape, img_shape=(512,512,1)).build()
# decoder.summary()


class Discriminator(tf.keras.Model):

    def __init__(self, f_extractor, img_shape=(512,512,1)):
        super().__init__()
        self.f_extractor = f_extractor
        self.img_shape = img_shape
    
    def build(self):
                
        f_input = Input(name='discriminator_input', shape=self.img_shape)
        f = self.f_extractor(f_input)
        f = GlobalAveragePooling2D()(f)
        judge = Dense(1, activation='sigmoid')(f)
        
        return tf.keras.Model(inputs=f_input, outputs=judge)

# f_shape = f_extractor.output.shape[1:]
# discriminator = Discriminator(f_shape=f_shape).build()
# discriminator.summary()

class GANomaly(tf.keras.Model):

    def __init__(self, INPUT_SHAPE=(512,512,1),loss_weights=[1,1,1]):
        super().__init__()
        self.INPUT_SHAPE = INPUT_SHAPE
        self.loss_weights = tf.Variable(loss_weights,dtype=tf.float32)
        self.g_encoder = Encoder(img_shape=self.INPUT_SHAPE,layer_name="g_encoder",GAP=True).build()
        self.g_decoder = Decoder(z_shape=self.g_encoder.output.shape[1:], img_shape=self.INPUT_SHAPE).build()
        self.e_encoder = Encoder(img_shape=self.INPUT_SHAPE,layer_name="e_encoder",GAP=True).build()
        self.f_extractor = Encoder(img_shape=self.INPUT_SHAPE,layer_name="f_extractor",GAP=False).build()
        
    def build(self):
        
        x_org = Input(name='main_input', shape=self.INPUT_SHAPE)
        z_org = self.g_encoder(x_org)
        x_hat = self.g_decoder(z_org)
        
        f_real, f_fake = self.f_extractor(x_org),self.f_extractor(x_hat) # AE.output
        z_hat = self.e_encoder(x_hat)
        
        # Define Model
        model = Model(inputs=x_org, outputs=[x_hat])
        
        # Define Loss function
        ADV_loss_v = self.loss_weights[0] * AdvLoss()(f_real,f_fake)
        CNT_loss_v = self.loss_weights[1] * CntLoss()(x_org,x_hat)
        ENC_loss_v = self.loss_weights[2] * EncLoss()(z_org,z_hat)
                
        model.add_loss(ADV_loss_v)
        model.add_loss(CNT_loss_v)
        model.add_loss(ENC_loss_v)

        model.add_metric(ADV_loss_v, name='adv_loss', aggregation='mean')
        model.add_metric(CNT_loss_v, name='cnt_loss', aggregation='mean')
        model.add_metric(ENC_loss_v, name='enc_loss', aggregation='mean')
        
        return model