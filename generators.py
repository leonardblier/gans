import numpy as np

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
    def __init__(self, shape_seed, shape_img, batch_size, model_g, real_data,
                 prob_g=0.5, label_gen=0.):
        self.shape_seed = shape_seed
        self.shape_img = shape_img
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
        X = np.zeros((self.batch_size,)+self.shape_img)
        y[:k] = self.label_gen

        # Generate pictures
        seed = np.random.normal(size=k*np.prod(self.shape_seed))
        seed = seed.reshape((k,)+self.shape_seed)
        X[:k] = self.model_g.predict_on_batch(seed)


        real_idx = np.random.randint(0,high=self.real_data.shape[0],
                                     size=self.batch_size - k)
        X[k:] = self.real_data[real_idx]
        return X, y
