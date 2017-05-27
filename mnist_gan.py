import numpy as np

from keras.layers import Input
from keras.layers.core import Dense, Flatten, Reshape, Lambda, Dropout
from keras.layers.convolutional import Conv2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard, EarlyStopping


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.datasets import mnist

from custom_keras import EarlyStoppingBound
from generators import generator_seed, generator_real_gen
from plottools import plot_metrics, plot_batch
from trainmethods import train_each, train_acc_bound

import ipdb

###############################
###### INPUTS PARAMETERS ######
shape_noise = (100,)
shape_img = (28,28, 1)
batch_size = 32
(mnist_train, y_train), (mnist_test, y_test) = mnist.load_data()
x_train = (mnist_train.astype('float32')  - 127.5) / 127.5
x_test = (mnist_test.astype('float32')  - 127.5) / 127.5
x_train = x_train.reshape((x_train.shape[0],) + shape_img)
x_test = x_test.reshape((x_test.shape[0],) + shape_img)
train_size = x_train.shape[0]
###############################



gen_seed_train = generator_seed(shape_noise, batch_size, label=1)
gen_seed_eval = generator_seed(shape_noise, batch_size, label=1)


###############################
##### Generator Model   #######
input_noise = Input(shape=shape_noise)
g = input_noise
g = Dense(1024, activation="relu")(g)
g = Dense(128*7*7, activation="relu")(g)
g = BatchNormalization()(g)

g = Reshape((7,7,128))(g)
#g = Dropout(0.5)(g)

# Input : (7,7), Output : (10,10)
g = UpSampling2D((2,2))(g)
g = Conv2D(64, (5,5), activation="relu")(g)
#g = Dropout(0.5)(g)

# Input : (10,10), Output : (16,16)
g = UpSampling2D((2,2))(g)
g = Conv2D(32, (5,5), activation="relu")(g)
#g = Dropout(0.5)(g)

# Input : (16,16), Output : (28,128)
g = UpSampling2D((2,2))(g)
g = Conv2D(1, (5,5), activation="tanh")(g)
#g = Lambda(lambda x: x[:,:,0])(g)
#g = Reshape(shape_img)(g)


model_g = Model(inputs=input_noise, outputs=g)
g_optim = SGD()
model_g.compile(g_optim, loss='mean_squared_error')
print("MODEL G :")
model_g.summary()

# Initialize the generator for generetaed/true img :
gen_real_gen = generator_real_gen(shape_noise, batch_size, model_g, 
                                  x_train, prob_g=.5, label_gen=0.)
gen_real_gen_eval = generator_real_gen(shape_noise, batch_size, model_g, 
                                       x_test, prob_g=.5, label_gen=0.)

##############################
#### Discriminator Model #####
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

        #d = ZeroPadding2D(padding=(2,2))(d)
        # d = self.conv3(d)
        # #d = LeakyReLU()(d)
        # d = AveragePooling2D()(d)

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



input_img_train = Input(shape=shape_img)
input_img_eval = Input(shape=shape_img)

model_discr_class = model_discriminator()
model_d_train = model_discr_class(input_img_train, dropout=True)
#optim_d_train = Adam(lr=0.00001)
#d_optim = SGD(lr=0.0001, momentum=0.9, nesterov=True)
d_optim = SGD(lr=0.0005, momentum=0.)
model_d_train.compile(d_optim, loss="binary_crossentropy",
                      metrics=["binary_accuracy"])

print("MODEL D :")
model_d_train.summary()

model_d_eval = model_discr_class(input_img_eval, dropout=False)
model_d_eval.trainable = False
for l in model_d_eval.layers:
    l.trainable = False

# These two lines are only there to avoid the batchnorm bug
#batch, y = gen_real_gen.next()
#model_d.train_on_batch(batch, y)

###############################
##### Comb of the two models ##

dg = model_d_eval(g)
model_dg = Model(inputs=input_noise, outputs=dg)
#dg_optim = Adam(lr=0.0005)
#dg_optim = SGD(lr=0.0001, momentum=0.9, nesterov=True)
dg_optim = SGD(lr=0.0005, momentum=0.)
model_dg.compile(optimizer=dg_optim, loss="binary_crossentropy",
                 metrics=["binary_accuracy"])





if __name__ == "__main__":
    train_each(model_d_train, model_dg, model_g,
               gen_real_gen, gen_real_gen_eval, gen_seed_train, gen_seed_eval,
               train_size, batch_size, nepoch=1000, 
               track_metrics=True, namefile_prefix="img/mnist4")
    # train_acc_bound(model_d_train, model_dg, gen_real_gen, gen_real_gen_eval,
    #                 gen_seed_train, gen_seed_eval, epochs=100, bound=0.8,
    #                 steps_per_cycle=1, validation_batchs=50)
