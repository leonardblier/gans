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


class generator_gen(object):
    def __init__(self, shape, batch_size, model, label=1.):
        self.shape = shape
        self.batch_size = batch_size
        self.label = label
        self.model = model

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        X = np.random.normal(size=np.prod(self.shape)*self.batch_size)
        X = X.reshape((self.batch_size,)+self.shape)
        y = np.full((self.batch_size,), self.label)
        
        X = self.model.predict_on_batch(X)
        return X, y #, list(f[:,0])

class generator_real_gen(object):
    def __init__(self, shape_seed, batch_size, model_g, real_data,
                 prob_g=0.5, label_gen=0.):
        self.shape_seed = shape_seed
        
        self.batch_size = batch_size
        self.model_g = model_g
        self.prob_g = prob_g
        self.label_gen = label_gen
        self.real_data = real_data
        self.shape_img = real_data.shape[1:]
        self.mode = None


    def __iter__(self):
        return self

    def __next__(self):
    	return self.next()

    def next(self):
        if self.mode == None:
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
            X[k:] = self.real_data[real_idx,:,:]
        if self.mode == "real":
            #y = np.full((self.batch_size,), 1 - self.label_gen)
            #real_idx = np.random.randint(0,high=self.real_data.shape[0],
            #                             size=self.batch_size)
            #X = self.real_data[real_idx,:,:]
            return self.get_batch_real()
        if self.mode == "gen":
            #y = np.full((self.batch_size,), self.label_gen)
            #seed = np.random.normal(size=self.batch_size*np.prod(self.shape_seed))
            #seed = seed.reshape((self.batch_size,)+self.shape_seed)
            #X = self.model_g.predict_on_batch(seed)
            return self.get_batch_gen()
        return X, y
    
    def get_batch_real(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        y = np.full((batch_size,), 1 - self.label_gen)
        real_idx = np.random.randint(0,high=self.real_data.shape[0],
                                     size=batch_size)
        X = self.real_data[real_idx,:,:]

        return X, y
    
    def get_batch_gen(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        y = np.full((batch_size,), self.label_gen)
        seed = np.random.normal(size=batch_size*np.prod(self.shape_seed))
        seed = seed.reshape((batch_size,)+self.shape_seed)
        X = self.model_g.predict_on_batch(seed)
        return X, y
    
    def set_mode(self, mode):
        self.mode = mode
        
    
    
