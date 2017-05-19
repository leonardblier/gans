import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
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

def preprocessing_dataset(folder, shape_img, crop_size, output_file):
    print("Preprocessing dataset")
    file_list = [os.path.join(folder, f) for f in os.listdir(folder)]
    X = np.zeros((len(file_list),)+shape_img , dtype=np.uint8)
    img_cropx, img_cropy = crop_size
    for i, f in enumerate(file_list):
        if i % 100 == 0:
            print("%d files over %d" % (i,len(file_list)))
        img = Image.open(os.path.join(folder, f))
        imgx, imgy = img.size
        img = img.crop(box=((imgx - img_cropx)//2,
                            (imgy - img_cropy)//2,
                            (imgx + img_cropx)//2,
                            (imgy + img_cropy)//2))
        img = img.resize(shape_img[:2])
        X[i] = np.array(img)
    with open(output_file, "wb") as file_dataset:
        np.save(file_dataset, X)
    return X
        
        
def load_data_celeba(numpy_file="celeba.npy", 
                     folder="/users/data/blier/img_align_celeba/"):
    if os.path.exists(numpy_file):
        with open(numpy_file, "rb") as f:
            X = np.load(f)
    else:
        X = preprocessing_dataset(folder, shape_img, (128,128),numpy_file)
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    return X_train, X_test
    
    
###############################
###### INPUTS PARAMETERS ######
shape_noise = (100,)
shape_img = (64,64,3)
batch_size = 32
celeba_train, celeba_test = load_data_celeba()
x_train = (celeba_train.astype(float)  - 127.5) / 127.5
x_test = (celeba_test.astype(float)  - 127.5) / 127.5
train_size = x_train.shape[0]
###############################



gen_seed_train = generator_seed(shape_noise, batch_size, label=1)
gen_seed_eval = generator_seed(shape_noise, batch_size, label=1)



###############################
##### Generator Model   #######
input_noise = Input(shape=shape_noise)
g = input_noise
#g = Dense(1024, activation="relu")(g)
g = Dense(4*4*1024, activation="relu")(g)
g = BatchNormalization()(g)
g = Reshape((4,4,1024))(g)
#g = Dropout(0.5)(g)

# Input : (4,4)
g = UpSampling2D((2,2))(g)
g = ZeroPadding2D(2)(g)
g = Conv2D(512, (5,5), activation="relu")(g)
#g = Dropout(0.5)(g)

# Input : (8,8)
g = UpSampling2D((2,2))(g)
g = ZeroPadding2D(2)(g)
g = Conv2D(256, (5,5), activation="relu")(g)
#g = Dropout(0.5)(g)

# Input : (16,16)
g = UpSampling2D((2,2))(g)
g = ZeroPadding2D(2)(g)
g = Conv2D(128, (5,5), activation="relu")(g)

# Input : (32,32)
g = UpSampling2D((2,2))(g)
g = ZeroPadding2D(2)(g)
g = Conv2D(3, (5,5), activation="tanh")(g)


#g = Lambda(lambda x: x[:,:,0])(g)
#g = Reshape(shape_img)(g)


model_g = Model(inputs=input_noise, outputs=g)
g_optim = SGD()
model_g.compile(g_optim, loss='mean_squared_error')


# Initialize the generator for generetaed/true img :
gen_real_gen = generator_real_gen(shape_noise, shape_img, batch_size,
                                  model_g, x_train, prob_g=.5, label_gen=0.)
gen_real_gen_eval = generator_real_gen(shape_noise, shape_img, batch_size,
                                       model_g, x_test, prob_g=.5,
                                       label_gen=0.)

print("SUMMARY FOR MODEL G")
model_g.summary()
##############################
#### Discriminator Model #####
class model_discriminator(object):
    def __init__(self):
        self.conv1 = Conv2D(128, (5,5))
        self.conv2 = Conv2D(256, (5,5))
        self.conv3 = Conv2D(512, (5,5))
        self.conv4 = Conv2D(1024, (5,5))

        self.dense1 = Dense(100)
        self.dense2 = Dense(1, activation="sigmoid")
        self.batchnorm = BatchNormalization()

    def __call__(self, input_img, dropout=False):
        d = input_img
        # d = Flatten()(input_img)
        # d = self.batchnorm(d)
        # d = Reshape(shape_img+(1,))(d)

        # CONV 1 : Input size 64
        d = ZeroPadding2D(padding=(2,2))(d)
        d = self.conv1(d)
        d = LeakyReLU()(d)
        d = AveragePooling2D()(d)

        if dropout:
           d = Dropout(0.5)(d)
        
        # CONV 2 : Input size 32
        d = ZeroPadding2D(padding=(2,2))(d)
        d = self.conv2(d)
        d = LeakyReLU()(d)
        d = AveragePooling2D()(d)
        if dropout:
           d = Dropout(0.5)(d)

    
        # Conv 3 : Input size 16
        d = ZeroPadding2D(padding=(2,2))(d)
        d = self.conv3(d)
        d = LeakyReLU()(d)
        d = AveragePooling2D()(d)
        
        # Conv 3 : Input size 8
        d = ZeroPadding2D(padding=(2,2))(d)
        d = self.conv4(d)
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



input_img_train = Input(shape=shape_img)
input_img_eval = Input(shape=shape_img)

model_discr_class = model_discriminator()
model_d_train = model_discr_class(input_img_train, dropout=True)
#optim_d_train = Adam(lr=0.00001)
d_optim = SGD(lr=0.0001, momentum=0.9, nesterov=True)
model_d_train.compile(d_optim, loss="binary_crossentropy",
                      metrics=["binary_accuracy"])


print("SUMMARY FOR MODEL D")
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
dg_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
model_dg.compile(optimizer=dg_optim, loss="binary_crossentropy",
                 metrics=["binary_accuracy"])




                          

if __name__ == "__main__":
    
    train_each(model_d_train, model_dg, model_g,
               gen_real_gen, gen_real_gen_eval, gen_seed_train, gen_seed_eval,
               train_size, batch_size, x_test[:64], nepoch=100, track_metrics=False)
    # train_acc_bound(model_d_train, model_dg, gen_real_gen, gen_real_gen_eval,
    #                 gen_seed_train, gen_seed_eval, epochs=100, bound=0.8,
    #                 steps_per_cycle=1, validation_batchs=50)
