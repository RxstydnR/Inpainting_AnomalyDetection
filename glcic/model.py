import tensorflow as tf
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Concatenate, Dropout, Input, Flatten, Dense
from tensorflow.keras.layers import ZeroPadding2D,Conv2DTranspose,LeakyReLU
from tensorflow.keras.layers import Conv2DTranspose, Concatenate
from tensorflow.keras.layers import Reshape, Lambda
from tensorflow.keras.models import Model

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Reference URL: https://github.com/ryanchao2012/glcic-keras/blob/master/models.py

def conv2d(x,filters,kernel_size,strides=1):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def dilated_conv2d(x,filters,kernel_size,dilation_rate,strides=1):
    x = Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,dilation_rate=dilation_rate,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def deconv2d(x,filters,kernel_size,strides=1):
    x = Conv2DTranspose(filters=filters,kernel_size=kernel_size, strides=strides,padding='same')(x)
    x - BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


class AdvLoss(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AdvLoss, self).__init__(**kwargs)
    
    def call(self, true, pred):
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(true,pred))
        return loss

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)


class ReconstLoss(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ReconstLoss, self).__init__(**kwargs)

    def call(self, true, pred):
        return tf.reduce_mean(tf.square(true-pred))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)


class CompletionNetwork(object):
    def __init__(self,input_shape=(256,256,3), mask_shape=(128,128,3)):
        self.input_shape = input_shape
        self.mask_shape = mask_shape

    def build(self):
        
        # An image in which the mask portion is filled with a certain color (simply black).
        input_masked_image = Input(shape=self.input_shape, dtype='float32')
        input_binary_mask = Input(shape=self.input_shape[:-1], dtype='float32')
        
        # Encode Convs
        x = conv2d(input_masked_image,filters=64,kernel_size=5,strides=1)
        x = conv2d(x,filters=128,kernel_size=3,strides=2)
        x = conv2d(x,filters=128,kernel_size=3,strides=1)
        x = conv2d(x,filters=256,kernel_size=3,strides=2)
        x = conv2d(x,filters=256,kernel_size=3,strides=1)
        x = conv2d(x,filters=256,kernel_size=3,strides=1)

        # Dilated Convs 
        x = dilated_conv2d(x,filters=256,kernel_size=3,dilation_rate=2)
        x = dilated_conv2d(x,filters=256,kernel_size=3,dilation_rate=4)
        x = dilated_conv2d(x,filters=256,kernel_size=3,dilation_rate=8)
        x = dilated_conv2d(x,filters=256,kernel_size=3,dilation_rate=16)

        # Convs
        x = conv2d(x,filters=256,kernel_size=3,strides=1)
        x = conv2d(x,filters=256,kernel_size=3, strides=1)

        # Decode Convs
        x = deconv2d(x,filters=128,kernel_size=4,strides=2)
        x = conv2d(x,filters=128, kernel_size=3, strides=1)
        x = deconv2d(x,filters=64,kernel_size=4,strides=2)
        x = conv2d(x,filters=32, kernel_size=3, strides=1)
        x = Conv2D(filters=3, kernel_size=3,strides=1,padding='same')(x)
        x = Activation('tanh')(x) # I am using tanh here
        output_image = x 

        # Cut out with mask area and merge with correct answer data.
        mask = Reshape((self.input_shape[0], self.input_shape[1],1))(input_binary_mask)
        outputs = Lambda(lambda x: x[0]*x[2] + x[1]*(1-x[2]))([output_image, input_masked_image, mask])

        model = Model(inputs=[input_masked_image,input_binary_mask], outputs=outputs)
        return model


class Discriminator(object):
    def __init__(self,input_shape=(256,256,3),mask_shape=(128,128,3)):
        self.input_shape = input_shape
        self.mask_shape = mask_shape

    def build(self):

        global_discriminator = self.build_global_discriminator()
        local_discriminator = self.build_local_discriminator()

        x = Concatenate()([global_discriminator.output,local_discriminator.output])
        outputs = Dense(1,activation="sigmoid")(x)
        
        model = Model(inputs=[global_discriminator.input,local_discriminator.input],outputs=outputs)
        return model

    def build_global_discriminator(self):
        
        input_image = Input(shape=self.input_shape, name='input_real_global', dtype='float32')
        
        x = conv2d(input_image, filters=64,kernel_size=5, strides=2)
        x = conv2d(x, filters=128, kernel_size=5, strides=2)
        x = conv2d(x, filters=256, kernel_size=5, strides=2)
        x = conv2d(x, filters=512, kernel_size=5, strides=2)
        x = conv2d(x, filters=512, kernel_size=5, strides=2)
        x = conv2d(x, filters=512, kernel_size=5, strides=2)
        x = Flatten(name='global_discriminator_flatten')(x)
        x = Dense(1024)(x)
        
        model = Model(inputs=input_image, outputs=x)
        return model

    def build_local_discriminator(self):

        input_patch = Input(shape=self.mask_shape, name='input_real_local', dtype='float32')

        x = conv2d(input_patch, filters=64, kernel_size=5, strides=2)
        x = conv2d(x, filters=128, kernel_size=5, strides=2)
        x = conv2d(x, filters=256, kernel_size=5, strides=2)
        x = conv2d(x, filters=512, kernel_size=5, strides=2)
        x = conv2d(x, filters=512, kernel_size=5, strides=2)
        x = Flatten(name='local_discriminator_flatten')(x)
        x = Dense(1024)(x)

        model = Model(inputs=input_patch, outputs=x)
        return model

    
class GLCIC(object):
    def __init__(self,input_shape=(256,256,3),mask_shape=(128,128,3)):
                 
        self.input_shape = input_shape
        self.mask_shape = mask_shape
        self.generator = CompletionNetwork(input_shape=self.input_shape,mask_shape=self.mask_shape).build()
        self.discriminator = Discriminator(input_shape=self.input_shape,mask_shape=self.mask_shape).build()
        self.model = self.build()
    
    def cropping(self, image, bbox):
        return tf.image.crop_to_bounding_box(image, bbox[1], bbox[0], bbox[3], bbox[2])
        
    def build(self):
    
        # Image filled with fixed color (simply black) mask part
        masked_image = self.generator.inputs[0]
        binary_mask = self.generator.inputs[1]
        fake_global = self.generator.outputs[0]

        real_global = self.discriminator.inputs[0]
        real_local = self.discriminator.inputs[1]

        # Genuine, let the discriminator evaluate the fake produced by the generator
        # local image is cut out from the mask area image and evaluated
        mask_area = Input(shape=(4,), name='input_mask_area', dtype='int32') # Mask area [x1,y1,x2,y2]

        cropping_layer = Lambda(
            lambda x: tf.map_fn(lambda z: self.cropping(z[0], z[1]), elems=x, dtype=tf.float32),
            output_shape=self.mask_shape, name='cropping_local')

        real_local = cropping_layer([real_global, mask_area])
        fake_local = cropping_layer([fake_global, mask_area])

        prob_real = self.discriminator([real_global, real_local])
        prob_fake = self.discriminator([fake_global, fake_local])
        
        # Combine them with real images
        probs = tf.concat([prob_real, prob_fake], axis=0, name='concat_probs')
                
        model = Model(inputs=[masked_image, binary_mask, mask_area, real_global], outputs=[probs,fake_global])

        n_real = tf.shape(prob_real)[0]
        n_fake = tf.shape(prob_fake)[0]

        labels = tf.concat([tf.ones(n_real,1), tf.zeros(n_fake,1)], axis=0)
        # labels += 0.05 * tf.random.uniform(tf.shape(labels)) # trick

        adv_loss = AdvLoss()(labels,probs)
        model.add_loss(adv_loss)
        model.add_metric(adv_loss, name='adv_loss', aggregation='mean')

        rec_loss = ReconstLoss()(real_global,fake_global)
        model.add_loss(rec_loss)
        model.add_metric(rec_loss, name='rec_loss', aggregation='mean')

        return model