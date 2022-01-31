import tensorflow as tf
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Concatenate, Input, Flatten, Dense
from tensorflow.keras.layers import Conv2DTranspose, Concatenate, UpSampling2D, LeakyReLU
from tensorflow.keras.layers import Reshape, Lambda
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
from tensorflow_addons.layers import SpectralNormalization
from tensorflow.keras.layers.experimental.preprocessing import Resizing

# Reference URL: https://github.com/ryanchao2012/glcic-keras/blob/master/models.py

def GatedConv2D(x,filters,ksize,stride=1,drate=1,padding='SAME',activation=tf.nn.elu,name=""):
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(drate*(ksize-1)/2)
        x = tf.pad(x, [[0,0],[p, p],[p, p],[0,0]], mode=padding)
        padding = 'VALID'
    x = Conv2D(filters, ksize, stride, dilation_rate=drate, activation=None, padding=padding, name=name)(x)

    if (filters == 3) or (activation is None):
        return x # conv for output
    
    x, y = tf.split(x, 2, 3)
    x = activation(x)
    y = tf.nn.sigmoid(y)
    x = x * y
    
    return x

def GatedDeconv2D(x,filters,padding='SAME',name=""):
    
    x = UpSampling2D(size=(2,2), interpolation="nearest")(x)
    x = GatedConv2D(x, filters, ksize=3, stride=1, padding=padding,name=name)
    
    return x

def SpectralNormalizationConv2D(x, filters, ksize=5, stride=2, name='conv', training=True):
    
    # x = conv2d_spectral_norm(x, filters, ksize, stride, 'SAME', name=name)
    x = SpectralNormalization(Conv2D(filters, ksize, stride, 'SAME', name=name))(x)
    x = LeakyReLU()(x)
    return x

def resize_mask_like(mask, size):
    
    height, width = size[0],size[1]
    mask_resized = Resizing(height, width, interpolation="nearest")(mask)

    return mask_resized

def resize_scale(x,scale):
    height, width = x.get_shape()[1:3] 
    height = int(height*scale)
    width = int(width*scale)
    
    x_resized = Resizing(height,width, interpolation="nearest")(x)
    return x_resized

def resize_shape(x,to_shape):
    x_resized = Resizing(to_shape[0],to_shape[1], interpolation="nearest")(x)
    return x_resized


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
        return tf.reduce_mean(tf.abs(true-pred))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)


class DeepFillV2(tf.keras.Model):
    """ DeepFillV2 
        Note: Sketch channel is not inplemented in this DeepFillV2
        Note: Contextual Attention is not inplemented in this DeepFillV2
              # Code of Contextual Attention, see https://github.com/JiahuiYu/generative_inpainting/blob/master/inpaint_ops.py
    """
    def __init__(self,img_shape,mask_shape):
        super().__init__('DeepFillV2')
        self.img_shape = img_shape
        self.mask_shape = mask_shape
        self.training = True
        self.filters = 48
        
        self.CoarseNetwork = self.build_CoarseNetwork()
        self.RefinementNetwork = self.build_RefinementNetwork()
        self.Discriminator = self.build_Discriminator()
        
    def build(self):
        
        true = Input(shape=self.img_shape, name="True_Input")
        masked = Input(shape=self.img_shape, name="Masked_Input")
        mask = Input(shape=self.mask_shape, name="Mask_Input")
        
        x_stage1 = self.CoarseNetwork([masked,mask])
        x_stage2 = self.RefinementNetwork([true,x_stage1,mask])
        
        prob_real = self.Discriminator(true)
        prob_fake = self.Discriminator(x_stage2)
        
        # return x_stage1, x_stage2, offset_flow # offset_flowはAttention is useful for visualization
        model = Model([true, masked, mask],[x_stage1, x_stage2, prob_fake])
            
        patch_probs = tf.concat([prob_real, prob_fake], axis=0, name='concat_probs')
        patch_labels = tf.concat([tf.ones(tf.shape(prob_real)), tf.zeros(tf.shape(prob_fake))], axis=0)

        adv_loss = AdvLoss()(patch_labels,patch_probs)
        model.add_loss(adv_loss)
        model.add_metric(adv_loss, name='adv_loss', aggregation='mean')

        # rec_loss1 = ReconstLoss()(true,x_stage1)
        # model.add_loss(rec_loss1)
        # model.add_metric(rec_loss1, name='rec_loss1', aggregation='mean')

        rec_loss = ReconstLoss()(true,x_stage2)
        model.add_loss(rec_loss)
        model.add_metric(rec_loss, name='rec_loss', aggregation='mean')

        
        return model
        
    def build_CoarseNetwork(self):

        filters = self.filters
        
        masked = Input(shape=self.img_shape, name="Masked_Input_Stage1")
        mask = Input(shape=self.mask_shape, name="Mask_Input_Stage1")
        x = Concatenate(axis=3)([masked,mask])
        
        x = GatedConv2D(x, filters, 5, 1, name='conv1')
        x = GatedConv2D(x, 2*filters, 3, 2, name='conv2_downsample')
        x = GatedConv2D(x, 2*filters, 3, 1, name='conv3')
        x = GatedConv2D(x, 4*filters, 3, 2, name='conv4_downsample')
        x = GatedConv2D(x, 4*filters, 3, 1, name='conv5')
        x = GatedConv2D(x, 4*filters, 3, 1, name='conv6')

        self.mask_reshape_size = x.get_shape()[1:3]
        
        x = GatedConv2D(x, 4*filters, 3, drate=2, name='conv7_atrous')
        x = GatedConv2D(x, 4*filters, 3, drate=4, name='conv8_atrous')
        x = GatedConv2D(x, 4*filters, 3, drate=8, name='conv9_atrous')
        x = GatedConv2D(x, 4*filters, 3, drate=16, name='conv10_atrous')
        x = GatedConv2D(x, 4*filters, 3, 1, name='conv11')
        x = GatedConv2D(x, 4*filters, 3, 1, name='conv12')
        x = GatedDeconv2D(x, 2*filters, name='conv13_upsample')
        x = GatedConv2D(x, 2*filters, 3, 1, name='conv14')
        x = GatedDeconv2D(x, filters, name='conv15_upsample')
        x = GatedConv2D(x, filters//2, 3, 1, name='conv16')
        x = GatedConv2D(x, 3, 3, 1, activation=None, name='conv17')
        x = Activation("tanh")(x)

        # 変更後
        # x = Lambda(lambda y: y[0]*y[2] + y[1]*(1.-y[2]))([x, true, mask])

        x_stage1 = x
        
        model = Model([masked,mask],x_stage1)
        
        return model
    
    def build_RefinementNetwork(self):
        
        true = Input(shape=self.img_shape, name="True_Input_Stage2")
        xin = Input(shape=self.img_shape, name="Coarse_Input_Stage2")
        mask = Input(shape=self.mask_shape, name="Mask_Input_Stage2")
        
        x = Lambda(lambda x: x[0]*x[2] + x[1]*(1.-x[2]))([xin, true, mask])
        
        # conv branch
        # xnow = tf.concat([x, ones_x, ones_x*mask], axis=3)
        x_coarse = tf.concat([x,mask], axis=3)
        
        filters = self.filters
        
        # Dilated Conv Branch
        x = GatedConv2D(x_coarse, filters, 5, 1, name='xconv1')
        x = GatedConv2D(x, 1*filters, 3, 2, name='xconv2_downsample')
        x = GatedConv2D(x, 2*filters, 3, 1, name='xconv3')
        x = GatedConv2D(x, 2*filters, 3, 2, name='xconv4_downsample')
        x = GatedConv2D(x, 4*filters, 3, 1, name='xconv5')
        x = GatedConv2D(x, 4*filters, 3, 1, name='xconv6')
        x = GatedConv2D(x, 4*filters, 3, drate=2, name='xconv7_atrous')
        x = GatedConv2D(x, 4*filters, 3, drate=4, name='xconv8_atrous')
        x = GatedConv2D(x, 4*filters, 3, drate=8, name='xconv9_atrous')
        x = GatedConv2D(x, 4*filters, 3, drate=16, name='xconv10_atrous')
        x_hallu = x
        
        # Contextual Attention Branch
        x = GatedConv2D(x_coarse, filters, 5, 1, name='pmconv1')
        x = GatedConv2D(x, 1*filters, 3, 2, name='pmconv2_downsample')
        x = GatedConv2D(x, 2*filters, 3, 1, name='pmconv3')
        x = GatedConv2D(x, 4*filters, 3, 2, name='pmconv4_downsample')
        x = GatedConv2D(x, 4*filters, 3, 1, name='pmconv5')
        x = GatedConv2D(x, 4*filters, 3, 1, name='pmconv6', activation=tf.nn.relu)
        
        # mask_s = resize_mask_like(mask, self.mask_reshape_size)
        # x, offset_flow = ContextualAttention(x, x, mask_s, 3, 1, rate=2)
        
        x = GatedConv2D(x, 4*filters, 3, 1, name='pmconv9')
        x = GatedConv2D(x, 4*filters, 3, 1, name='pmconv10')
        pm = x
        
        # Inpainting Upsampling
        x = tf.concat([x_hallu, pm], axis=3)
        x = GatedConv2D(x, 4*filters, 3, 1, name='allconv11')
        x = GatedConv2D(x, 4*filters, 3, 1, name='allconv12')
        x = GatedDeconv2D(x, 2*filters, name='allconv13_upsample')
        x = GatedConv2D(x, 2*filters, 3, 1, name='allconv14')
        x = GatedDeconv2D(x, filters, name='allconv15_upsample')
        x = GatedConv2D(x, filters//2, 3, 1, name='allconv16')
        x = GatedConv2D(x, 3, 3, 1, activation=None, name='allconv17')
        x = Activation("tanh")(x)

        # x = Lambda(lambda y: y[0]*y[2] + y[1]*(1.-y[2]))([x, true, mask])
        x_stage2 = x
        
        model = Model([true,xin,mask],x_stage2)
        
        return model
    
    def build_Discriminator(self):
        
        xin = Input(shape=self.img_shape, name="Image_Input_Discriminator")
        
        filters = 64
        x = SpectralNormalizationConv2D(xin, filters, name='d_conv1', training=self.training)
        x = SpectralNormalizationConv2D(x, filters*2, name='d_conv2', training=self.training)
        x = SpectralNormalizationConv2D(x, filters*4, name='d_conv3', training=self.training)
        x = SpectralNormalizationConv2D(x, filters*4, name='d_conv4', training=self.training)
        x = SpectralNormalizationConv2D(x, filters*4, name='d_conv5', training=self.training)
        x = SpectralNormalizationConv2D(x, filters*4, name='d_conv6', training=self.training)
        x = Flatten(name='flatten')(x)
        patch_out = Activation('sigmoid', name='d_output')(x)
        
        model = Model(xin, patch_out)
        return model




# 変更前
# xは復元した場所以外は元の綺麗な画像を取得する
# x = (xin * mask) + (img[:, :, :, 0:3] * (1.-mask)) # 綺麗な領域
# x.set_shape(img[:, :, :, 0:3].get_shape().as_list())

##################################################
# 以下 Contectual Attentionの実装
##################################################
# def flow_to_image(flow):
#     """Transfer flow map to image.
#     Part of code forked from flownet.
#     """
#     out = []
#     maxu = -999.
#     maxv = -999.
#     minu = 999.
#     minv = 999.
#     maxrad = -1
#     for i in range(flow.shape[0]):
#         u = flow[i, :, :, 0]
#         v = flow[i, :, :, 1]
#         idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
#         u[idxunknow] = 0
#         v[idxunknow] = 0
#         maxu = max(maxu, np.max(u))
#         minu = min(minu, np.min(u))
#         maxv = max(maxv, np.max(v))
#         minv = min(minv, np.min(v))
#         rad = np.sqrt(u ** 2 + v ** 2)
#         maxrad = max(maxrad, np.max(rad))
#         u = u/(maxrad + np.finfo(float).eps)
#         v = v/(maxrad + np.finfo(float).eps)
#         img = compute_color(u, v)
#         out.append(img)
#     return np.float32(np.uint8(out))

# def flow_to_image_tf(flow, name='flow_to_image'):
#     """Tensorflow ops for computing flow to image.
#     """
#     img = tf.py_function(flow_to_image, [flow], Tout=tf.float32)#, stateful=False)
#     img.set_shape(flow.get_shape().as_list()[0:-1]+[3])
#     img = img / 127.5 - 1.
#     return img

# def ContextualAttention(f, b, mask=None, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10., training=True, fuse=True):
#     """ Contextual attention layer implementation.
#         Contextual attention is first introduced in publication:
#             Generative Image Inpainting with Contextual Attention, Yu et al.
#         Args:
#             f: Input feature to match (foreground).
#             b: Input feature for match (background).
#             mask: Input mask for t, indicating patches not available.
#             ksize: Kernel size for contextual attention.
#             stride: Stride for extracting patches from t.
#             rate: Dilation for matching.
#             softmax_scale: Scaled softmax for attention.
#             training: Indicating if current graph is training or inference.
#         Returns:
#             tf.Tensor: output
#     """
#     # get shapes
#     f_shape = tf.shape(f)
#     b_shape = tf.shape(b)
#     f_int_shape = f.get_shape().as_list() # -> [None, 64, 64, 96]
#     b_int_shape = b.get_shape().as_list() # -> [None, 64, 64, 96]
    
            
#     # extract patches from background with stride and rate
#     kernel = 2*rate
#     raw_w = tf.image.extract_patches(
#         b, [1,kernel,kernel,1], [1,rate*stride,rate*stride,1], [1,1,1,1], padding='SAME') # raw_w -> (None, 32, 32, 1536)
        
#     raw_w = tf.reshape(raw_w, [b_shape[0], -1, kernel, kernel, b_shape[3]]) # [Batch_size,Patch_num,Patch_size,Patch_size,channel]
#     raw_w = tf.transpose(raw_w, [0,2,3,4,1])  # transpose to b*k*k*c*hw
    
#     # downscaling foreground option: downscaling both foreground and
#     # background for matching and use original background for reconstruction.
#     f = resize_scale(f, scale=1./rate)
#     b = resize_shape(b, to_shape=[int(b_int_shape[1]/rate), int(b_int_shape[2]/rate)])
#     if mask is not None:
#         mask = resize_scale(mask, scale=1./rate)
        
#     f_shape = tf.shape(f)
#     f_int_shape = f.get_shape().as_list() # [None, 32, 32, 96]
    
#     # 変更前
#     # from t(H*W*C) to w(b*k*k*c*h*w)
#     f_ = tf.split(f, f_int_shape[1], axis=0)
#     print(len(f_))
#     print(f_[0])
    
#     assert False
        
#     # f_groups = tf.split(f, f_int_shape[0], axis=0)
#     # 変更後
#     # f_groups = tf.split(f, int_fs[1], axis=0)
#     # f_groups = tf.reshape(f,[f_shape[0],f_int_shape[1]*f_int_shape[2]*f_int_shape[3]])
#     # f_groups = tf.reshape(f,[f_shape[0],-1])
    
#     b_shape = tf.shape(b)
#     b_int_shape = b.get_shape().as_list()
#     w = tf.image.extract_patches(
#         b, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
#     w = tf.reshape(w, [f_shape[0], -1, ksize, ksize, f_shape[3]]) # fのshapeで良いの?? bのshapeと比較が必要...（一緒だったので混同している可能性あり）
#     w = tf.transpose(w, [0,2,3,4,1])  # transpose to b*k*k*c*hw
        
#     # process mask
#     if mask is None:
#         mask = tf.zeros([1, b_shape[1], b_shape[2], 1])
#     m = tf.image.extract_patches(
#         mask, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
#     m = tf.reshape(m, [1, -1, ksize, ksize, 1])
#     m = tf.transpose(m, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
#     m = m[0]
#     mm = tf.cast(tf.equal(tf.reduce_mean(m, axis=[0,1,2], keepdims=True), 0.), tf.float32)
    
#     # 変更前
#     #w_groups = tf.split(w, b_int_shape[0], axis=0)
#     #raw_w_groups = tf.split(raw_w, b_int_shape[0], axis=0)
    
#     # 変更後
#     # f_groups = tf.reshape(f,[f_shape[0],f_int_shape[1]*f_int_shape[2]*f_int_shape[3]])
#     # w_groups = tf.reshape(w,[b_shape[0],b_int_shape[1]*b_int_shape[2]*b_int_shape[3]])
#     # raw_w_groups = tf.reshape(raw_w,[b_shape[0],b_int_shape[1]*b_int_shape[2]*b_int_shape[3]])

#     y = []
#     offsets = []
#     k = fuse_k
#     scale = softmax_scale
#     fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])
    
#     for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
#         # conv for compare
#         wi = wi[0]
#         wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0,1,2])), 1e-4)
#         yi = tf.nn.conv2d(xi, wi_normed, strides=[1,1,1,1], padding="SAME")

#         # conv implementation for fuse scores to encourage large patches
#         if fuse:
#             yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
#             yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
#             yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1], bs[2]])
#             yi = tf.transpose(yi, [0, 2, 1, 4, 3])
#             yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
#             yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
#             yi = tf.reshape(yi, [1, fs[2], fs[1], bs[2], bs[1]])
#             yi = tf.transpose(yi, [0, 2, 1, 4, 3])
#         yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1]*bs[2]])

#         # softmax to match
#         yi *=  mm  # mask
#         yi = tf.nn.softmax(yi*scale, 3)
#         yi *=  mm  # mask

#         offset = tf.argmax(yi, axis=3, output_type=tf.int32)
#         offset = tf.stack([offset // fs[2], offset % fs[2]], axis=-1)
        
#         # deconv for patch pasting
#         # 3.1 paste center
#         wi_center = raw_wi[0]
#         yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_fs[1:]], axis=0), strides=[1,rate,rate,1]) / 4.
#         y.append(yi)
#         offsets.append(offset)
        
#     y = tf.concat(y, axis=0)
#     y.set_shape(raw_int_fs)
#     offsets = tf.concat(offsets, axis=0)
#     offsets.set_shape(int_bs[:3] + [2])
    
#     # case1: visualize optical flow: minus current position
#     h_add = tf.tile(tf.reshape(tf.range(bs[1]), [1, bs[1], 1, 1]), [bs[0], 1, bs[2], 1])
#     w_add = tf.tile(tf.reshape(tf.range(bs[2]), [1, 1, bs[2], 1]), [bs[0], bs[1], 1, 1])
#     offsets = offsets - tf.concat([h_add, w_add], axis=3)
    
#     # to flow image
#     flow = flow_to_image_tf(offsets)
    
#     # # case2: visualize which pixels are attended
#     # flow = highlight_flow_tf(offsets * tf.cast(mask, tf.int32))
#     if rate != 1:
#         flow = resize_scale(flow, scale=rate)
        
#     return y, flow

