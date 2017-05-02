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

import ipdb

(x_train, y_train), (x_test, y_test) = mnist.load_data()

###############################
###### INPUTS PARAMETERS ######
shape_noise = (100,)
shape_img = (28,28)
train_size = x_train.shape[0]
batch_size = 32
(mnist_train, y_train), (mnist_test, y_test) = mnist.load_data()
x_train = (mnist_train.astype(float)  - 127.5) / 127.5
###############################


###############################
#### Generators ###############
class generator_seed(object):
    def __init__(self, shape, batch_size, label=1.):
        self.shape = shape
        self.batch_size = batch_size
        self.label = label

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        X = np.random.normal(size=np.prod(self.shape)*self.batch_size)
        X = X.reshape((self.batch_size,)+self.shape)
        y = np.full((self.batch_size,), self.label)
        return X, y

class generator_real_gen(object):
    def __init__(self, shape_seed, batch_size, model_g, real_data,
                 prob_g=0.5, label_gen=0.):
        self.shape_seed = shape_seed
        self.batch_size = batch_size
        self.model_g = model_g
        self.prob_g = prob_g
        self.label_gen = label_gen
        self.real_data = real_data


    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        k = np.random.binomial(self.batch_size, self.prob_g)
        y = np.full((self.batch_size,), 1 - self.label_gen)
        X = np.zeros((self.batch_size,)+shape_img)
        y[:k] = self.label_gen

        # Generate pictures
        seed = np.random.normal(size=k*np.prod(self.shape_seed))
        seed = seed.reshape((k,)+self.shape_seed)
        X[:k] = self.model_g.predict_on_batch(seed)


        real_idx = np.random.randint(0,high=self.real_data.shape[0],
                                     size=self.batch_size - k)
        X[k:] = self.real_data[real_idx]
        return X, y


def plot_batch(batch, namefile=None):
    fig, axes = plt.subplots(4, 8,
                             subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in zip(range(32), axes.flat):
        ax.imshow(((batch[i,:,:]+1)*(255./2)).astype(np.uint8),
                  interpolation=None,
                  cmap="gray")
    if namefile is None:
        plt.show()
    else:
        plt.savefig(namefile)
    plt.clf()
    plt.close()


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
g = Reshape(shape_img)(g)


model_g = Model(inputs=input_noise, outputs=g)
g_optim = SGD()
model_g.compile(g_optim, loss='mean_squared_error')


# Initialize the generator for generetaed/true img :
gen_real_gen = generator_real_gen(shape_noise, 32, model_g, x_train,
                                  prob_g=.5, label_gen=0.)
gen_real_gen_eval = generator_real_gen(shape_noise, 32, model_g, x_test,
                                       prob_g=.5, label_gen=0.)
##############################
#### Discriminator Model #####
class model_discriminator(object):
    def __init__(self):
        self.conv1 = Conv2D(64, (5,5), activation='relu')
        self.conv2 = Conv2D(128, (5,5), activation='relu')
        self.conv3 = Conv2D(256, (5,5), activation='relu')

        self.dense1 = Dense(1024, activation='relu')
        self.dense2 = Dense(1, activation="sigmoid")
        self.batchnorm = BatchNormalization()

    def __call__(self, input_img, dropout=False):
        d = input_img
        # d = Flatten()(input_img)
        # d = self.batchnorm(d)
        d = Reshape(shape_img+(1,))(d)

        # CONV 1
        #d = ZeroPadding2D(padding=(2,2))(d)
        d = self.conv1(d)
        #d = LeakyReLU()(d)
        d = AveragePooling2D()(d)

        if dropout:
           d = Dropout(0.5)(d)
        #d = ZeroPadding2D(padding=(2,2))(d)
        d = self.conv2(d)
        #d = LeakyReLU()(d)
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
model_d_train = model_discr_class(input_img_train, dropout=False)
#optim_d_train = Adam(lr=0.00001)
d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
model_d_train.compile(d_optim, loss="binary_crossentropy",
                      metrics=["binary_accuracy"])




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




def train_acc_bound():
    tboard_g = TensorBoard(log_dir='./logs', histogram_freq=1,
                           write_graph=True, write_images=True)
    tboard_d = TensorBoard(log_dir='./logs', histogram_freq=1,
                           write_graph=True, write_images=True)
    bound_acc = EarlyStoppingBound(0.7, 'val_binary_accuracy', 'upper')
    early_stop = EarlyStopping('val_loss', min_delta=0.01, patience=6)

    epoch_g = 0
    epoch_d = 0
    for count in range(100):
        print("Starting loop "+str(count))

        print("Train generative network")
        model_dg.fit_generator(gen_seed_train, steps_per_epoch=100,
                               epochs = 100,
                               callbacks=[tboard_g, bound_acc, early_stop],
                               validation_data=gen_seed_eval,
                               validation_steps=50)#,


        print("Printing an image.")
        batch_seed, _ = gen_seed_eval.next()
        batch_img = model_g.predict_on_batch(batch_seed)
        plot_batch(batch_img, namefile="img/gen"+str(count)+".png")



        print("Train Discriminative network")
        model_d_train.fit_generator(gen_real_gen, steps_per_epoch=100,
                                    epochs = 100, workers=1,
                                    callbacks=[bound_acc, early_stop],
                                    validation_data=gen_real_gen_eval,
                                    validation_steps=50)

def train_each():
    gamma = 1
    for epoch in range(100):
        print("EPOCH : "+str(epoch))
        d_loss, d_acc = 0., 0.
        dg_loss, dg_acc = 0., 0.
        for index in range(x_train.shape[0]//batch_size):
            if index % 20 == 1:
                print("Discriminator loss : %f accuracy : %f" % \
                      (d_loss / index, d_acc / index))
                print("Generator loss : %f accuracy : %f" % \
                      (dg_loss / (index*gamma), dg_acc / (index*gamma)))


            X, y = gen_real_gen.next()
            l, acc = model_d_train.train_on_batch(X, y)
            d_loss += l
            d_acc += acc

            for _ in range(gamma):
                X, y = gen_seed_train.next()
                l, acc = model_dg.train_on_batch(X,y)
                dg_loss += l
                dg_acc += acc

        print("Printing an image.")
        batch_seed, _ = gen_seed_eval.next()
        batch_img = model_g.predict_on_batch(batch_seed)
        plot_batch(batch_img, namefile="img/gen"+str(epoch)+".png")

if __name__ == "__main__":
    train_each()
