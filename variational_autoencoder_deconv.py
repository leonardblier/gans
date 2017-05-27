'''This script demonstrates how to build a variational autoencoder
with Keras and deconvolution layers.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer, Dropout
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.optimizers import SGD, Adam
from keras.datasets import mnist

from custom_keras import CustomVariationalLayer
from plottools import plot_latent_space, plot_manifold
from generators import generator_real_gen


############################################
###### PARAMETERS ##########################
############################################
# input image dimensions
img_rows, img_cols, img_chns = 28, 28, 1
# number of convolutional filters to use
filters = 64
# convolution kernel size
num_conv = 3

batch_size = 100
if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)
latent_dim = 2
intermediate_dim = 128
epsilon_std = 1.0
epochs = 10

(x_train, _), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape((x_test.shape[0],) + original_img_size)

print('x_train.shape:', x_train.shape)
##############################################
##############################################



##############################################
################## ENCODER ###################


batch_size = batch_size
original_img_size = original_img_size

#x = Input(batch_shape=(batch_size,) + original_img_size)
conv_1 = Conv2D(img_chns,
                kernel_size=(2, 2),
                padding='same', activation='relu')(x)
conv_2 = Conv2D(filters,
                kernel_size=(2, 2),
                padding='same', activation='relu',
                strides=(2, 2))(conv_1)
conv_3 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_2)
conv_4 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_3)
flat = Flatten()(conv_4)
hidden = Dense(intermediate_dim, activation='relu')(flat)

z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

#################################################
################## DECODER ######################

decoder_hid = Dense(intermediate_dim, activation='relu')
decoder_upsample = Dense(filters * 14 * 14, activation='relu')

if K.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters, 14, 14)
else:
    output_shape = (batch_size, 14, 14, filters)

decoder_reshape = Reshape(output_shape[1:])
decoder_deconv_1 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
decoder_deconv_2 = Conv2DTranspose(filters, num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
if K.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters, 29, 29)
else:
    output_shape = (batch_size, 29, 29, filters)
    
decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='valid',
                                          activation='relu')
decoder_mean_squash = Conv2D(img_chns,
                             kernel_size=2,
                             padding='valid',
                             activation='sigmoid')

hid_decoded = decoder_hid(z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)
#################################################



###################################################
########### MODEL VAE (and train) #################
y = CustomVariationalLayer(img_rows, img_cols)(\
    [x, x_decoded_mean_squash, z_mean, z_log_var])
vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()

vae.fit(x_train, shuffle=True, epochs=epochs, batch_size=batch_size,
        validation_data=(x_test, x_test))
###################################################


#################################################
######### MODEL ENCODER and GENERATOR ###########
encoder = Model(x, z_mean)

decoder_input = Input(shape=(latent_dim,))
_hid_decoded = decoder_hid(decoder_input)
_up_decoded = decoder_upsample(_hid_decoded)
_reshape_decoded = decoder_reshape(_up_decoded)
_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
generator = Model(decoder_input, _x_decoded_mean_squash)

#################################################
######## MODEL DISCRIMINATOR ####################
class model_discriminator(object):
    def __init__(self):
        self.conv1 = Conv2D(64, (5,5))
        self.conv2 = Conv2D(128, (5,5))
        self.conv3 = Conv2D(256, (5,5))

        self.dense1 = Dense(100)
        self.dense2 = Dense(1, activation="sigmoid")
        self.batchnorm = BatchNormalization()

    def __call__(self, input_img, dropout=False):
        d = input_img
        # d = Flatten()(input_img)
        # d = self.batchnorm(d)
        # d = Reshape(shape_img+(1,))(d)

        # CONV 1
        #d = ZeroPadding2D(padding=(2,2))(d)
        d = self.conv1(d)
        d = LeakyReLU()(d)
        d = AveragePooling2D()(d)

        if dropout:
           d = Dropout(0.5)(d)
        #d = ZeroPadding2D(padding=(2,2))(d)
        d = self.conv2(d)
        d = LeakyReLU()(d)
        d = AveragePooling2D()(d)
        if dropout:
           d = Dropout(0.5)(d)

        #d = MaxPooling2D((2,2))(d)

        d = ZeroPadding2D(padding=(2,2))(d)
        d = self.conv3(d)
        d = LeakyReLU()(d)
        d = AveragePooling2D()(d)

        d = Flatten()(d)
        if dropout:
            d = Dropout(0.5)(d)

        d = self.dense1(d)
        if dropout:
            d = Dropout(0.5)(d)
        #d = LeakyReLU()(d)
        d = self.dense2(d)

        model = Model(inputs=input_img, outputs=d)

        return model



input_img_train = Input(shape=original_img_size)
input_img_eval = Input(shape=original_img_size)

model_discr_class = model_discriminator()
discriminator_train = model_discr_class(input_img_train, dropout=True)
#optim_d_train = Adam(lr=0.00001)
d_optim = SGD(lr=0.001, momentum=0.9, nesterov=True)
discriminator_train.compile(d_optim, loss="binary_crossentropy",
                      metrics=["binary_accuracy"])

print("MODEL D :")
discriminator_train.summary()

discriminator_eval = model_discr_class(input_img_eval, dropout=False)
discriminator_eval.trainable = False
for l in discriminator_eval.layers:
    l.trainable = False

# These two lines are only there to avoid the batchnorm bug
#batch, y = gen_real_gen.next()
#model_d.train_on_batch(batch, y)

###############################
##### Comb of the two models ##
dg = discriminator_eval(_x_decoded_mean_squash)
model_dg = Model(inputs=decoder_input, outputs=dg)
#dg_optim = Adam(lr=0.0005)
dg_optim = SGD(lr=0.001, momentum=0.9, nesterov=True)
model_dg.compile(optimizer=dg_optim, loss="binary_crossentropy",
                 metrics=["binary_accuracy"])


##################################
#### GEN for training #####
gen_real_gen = generator_real_gen((latent_dim,), batch_size, generator, 
                                  x_train, prob_g=.5, label_gen=0.)
gen_real_gen_eval = generator_real_gen((latent_dim,), batch_size, 
                                       generator, x_test, prob_g=.5,
                                       label_gen=0.)
                                       
print("Test generators")
x, y = gen_real_gen.next()
                                       
discriminator_train.fit_generator(gen_real_gen, 60000//batch_size, epochs=20,
    validation_data=gen_real_gen_eval, validation_steps=10000//batch_size)


#################################################
#################### PLOTS ######################
#plot_latent_space(x_test, y_test, encoder, batch_size, 
#                  "img/vae_latent_space.png")
#plot_manifold(generator, batch_size)



    
