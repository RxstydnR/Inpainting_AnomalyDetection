import tensorflow as tf

class AdvLoss(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AdvLoss, self).__init__(**kwargs)
    
    def call(self, f_org, f_rec):
        return tf.reduce_mean(tf.square(f_org - tf.reduce_mean(f_rec, axis=0)))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)
    
    
class CntLoss(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CntLoss, self).__init__(**kwargs)

    def call(self, x_org, x_rec):
        # l1
        # return tf.reduce_mean(tf.abs(x_org - x_rec)) 
        # l2
        return tf.reduce_mean(tf.square(x_org - x_rec))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)
    
    
class EncLoss(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EncLoss, self).__init__(**kwargs)

    def call(self, z_org, z_rec):
        return tf.reduce_mean(tf.square(z_org - z_rec))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)