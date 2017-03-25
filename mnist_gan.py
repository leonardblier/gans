import numpy as np

from keras.layers import Input
from keras.layers.core import Dense, Flatten, Reshape, Lambda
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import SGD, Adam

import matplotlib.pyplot as plt

from keras.datasets import mnist

import ipdb

(x_train, y_train), (x_test, y_test) = mnist.load_data()

###############################
###### INPUTS PARAMETERS ######
shape_noise = (128,)
shape_img = (28,28)
train_size = x_train.shape[0]
batch_size = 32
(mnist_train, y_train), (mnist_test, y_test) = mnist.load_data()
x_train = mnist_train.astype(float)  / 255
x_train = (x_train * 2) - 1
###############################

###############################
##### Generator Model   #######
input_noise = Input(shape=shape_noise)
g = input_noise
g = BatchNormalization()(g)
g = Dense(128*7*7, activation="relu")(g)
g = Reshape((7,7,128))(g)

# Input : (7,7), Output : (10,10)
g = UpSampling2D((2,2))(g)
g = Conv2D(64, (5,5), activation="relu")(g)

# Input : (10,10), Output : (16,16)
g = UpSampling2D((2,2))(g)
g = Conv2D(32, (5,5), activation="relu")(g)

# Input : (16,16), Output : (28,128)
g = UpSampling2D((2,2))(g)
g = Conv2D(1, (5,5), activation="tanh")(g)

#g = Lambda(lambda x: x[:,:,0])(g)

model_g = Model(inputs=input_noise, outputs=g)
g_optim = SGD()
model_g.compile(g_optim, loss='mean_squared_error')


##############################
#### Discriminator Model #####
input_img = Input(shape=shape_img+(1,))
d = Reshape(shape_img+(1,))(input_img)
d = BatchNormalization()(d)

d = Conv2D(64, (5,5), strides=(2,2))(d)
d = LeakyReLU()(d)
#d = MaxPooling2D((2,2))(d)
d = Conv2D(32, (5,5), strides=(2,2))(d)
d = LeakyReLU()(d)
#d = MaxPooling2D((2,2))(d)

d = Flatten()(d)
d = Dense(512)(d)
d = LeakyReLU()(d)
d = Dense(1, activation="sigmoid")(d)


#d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
d_optim = Adam()
model_d = Model(inputs=input_img, outputs=d)
model_d.compile(d_optim,
                loss="binary_crossentropy",
                metrics=["binary_accuracy"])


###############################
##### Comb of the two models ##
model_d.trainable = False
for l in model_d.layers:
    l.trainable = False
dg = model_d(g)
model_dg = Model(inputs=input_noise, outputs=dg)
dg_optim = Adam()
model_dg.compile(optimizer=dg_optim, loss="binary_crossentropy",
                 metrics=["binary_accuracy"])


###############################
#### Generators ###############
class generator_seed(object):
    def __init__(self, shape, batch_size, label_noise=0):
        self.shape = shape
        self.batch_size = batch_size
        self.label_noise = label_noise

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        X = np.random.normal(size=np.prod(self.shape)*self.batch_size)
        X = X.reshape((self.batch_size,)+self.shape)
        y = np.full((self.batch_size,), self.label_noise)
        return X, y

class generator_real_gen(object):
    def __init__(self, shape_seed, batch_size, model_g, real_data,
                 prob_g=0.5, label_noise=1):
        self.shape_seed = shape_seed
        self.batch_size = batch_size
        self.model_g = model_g
        self.prob_g = prob_g
        self.label_noise = label_noise
        self.real_data = real_data


    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        y = np.random.binomial(1, self.prob_g, size=self.batch_size)
        y_bool = y.astype(bool)

        u = np.random.rand()

        if u < self.prob_g:
            seed = np.random.normal(size=np.prod(self.shape_seed)*self.batch_size)
            seed = seed.reshape((self.batch_size,)+self.shape_seed)
            X = self.model_g.predict_on_batch(seed)
            y = np.full((self.batch_size,), self.label_noise)
        else:
            real_idx = np.random.randint(0,high=train_size, size=self.batch_size)
            X = self.real_data[real_idx,:,:,np.newaxis]
            y = np.full((self.batch_size,), 1 - self.label_noise)
        return X, y

def plot_batch(batch, namefile=None):
    fig, axes = plt.subplots(4, 8)#, figsize=(12, 6),
    #                     subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.3, wspace=0.05)
    for i, ax in zip(range(32), axes.flat):
        ax.imshow(((batch[i,:,:,0]+1)*(255./2)).astype(np.uint8), interpolation=None,
                  cmap="gray")
    if namefile is None:
        plt.show()
    else:
        plt.savefig(namefile)
    plt.clf()
    plt.close()


#################################
#### Training ###################
gen_seed = generator_seed(shape_noise, 32)
gen_real_gen = generator_real_gen(shape_noise, 32, model_g, x_train)

count = 0
for _ in range(100):
    acc_g = 0.
    acc_d = 0

    while acc_g < 0.8:
        print("Train generative network")
        hist = model_dg.fit_generator(gen_seed, steps_per_epoch=10,
                                      epochs = 1)
        acc_g = hist.history["binary_accuracy"][-1]

    batch_seed, _ = gen_seed.next()
    batch_img = model_g.predict_on_batch(batch_seed)
    plot_batch(batch_img, namefile="img/gen"+str(count)+".png")
    while acc_d < 0.8:
        print("Train Discriminative network")
        hist = model_d.fit_generator(gen_real_gen, steps_per_epoch=10,
                                     epochs = 1)
        acc_d = hist.history["binary_accuracy"][-1]

    count += 1
