import numpy as np
#import pandas as pd
import tensorflow as tf
import datetime
from tensorflow import keras
from tensorflow.keras import layers
import json

import FEM
from utils import merge_first_dims

def test_linearity(model, input_shape_1, input_shape_2, output_length):
    n_in = input_shape_1 + input_shape_2
    def format(a): return [np.array([a[:input_shape_1]]), np.array([a[input_shape_1:]])]
    X = np.random.rand(n_in) -0.5#* 100 - 50
    Y_1 = model(format(X))
    I = np.identity(n_in)
    O = np.zeros(n_in)
    Y_2 = model(format(O))
    print('model(0):', Y_2)
    for i, x in enumerate(X):
        y = (model(format(I[i])) - model(format(O)))*x
        Y_2 += y
        print("{:0.7f}".format(np.linalg.norm(y) / np.linalg.norm(x)), end=' ')
    #print('Y_1:', Y_1)
    #print('Y_2:', Y_2)
    print('difference norm:', np.linalg.norm((Y_1 - Y_2).numpy()))


def lrelu(x, alpha=0.01):
    return tf.keras.activations.relu(x, alpha=alpha)#, threshold=0,  max_value=0.01)
    #return tf.keras.activations.relu(x, alpha=0.3)#, threshold=0,  max_value=0.01)

def get_DNN(input_shape, output_length, depth, bn_depth, lr, n_layers_1=1, n_layers_2=2, l1_penalty=0, activation=lrelu):

    L1_reg = keras.regularizers.L1(l1=l1_penalty)
    inputs = keras.Input(shape=input_shape)
    x = inputs

    for i in range(n_layers_1):
        x = layers.Dense(depth,activation=activation, kernel_regularizer=L1_reg)(x)
    x = layers.Dense(bn_depth,activation=activation, kernel_regularizer=L1_reg)(x)

    for i in range(n_layers_2):
        x = layers.Dense(depth,activation=activation, kernel_regularizer=L1_reg)(x)
    x = layers.Dense(output_length,kernel_regularizer=L1_reg)(x)

    model = keras.Model(inputs=inputs, outputs=x)
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=opt)
    return model

def get_pgLR(input_shape_1, input_shape_2, output_length, bn_depth, lr, l1_penalty=0):
    '''
    Fully connected nerual network with 2 inputs, one at the start and one at a bottleneck
    '''
    L1_reg = keras.regularizers.L1(l1=l1_penalty)

    x1 = keras.Input(shape=input_shape_1)
    x = layers.Dense(bn_depth,kernel_regularizer=L1_reg)(x1)
    model_1 = keras.Model(x1,x)

    x2 = keras.Input(shape=input_shape_2)
    model_2 = keras.Model(x2, x2)

    combined_input = layers.concatenate([model_1.output, model_2.output])
    x = combined_input 
    x = layers.Dense(output_length,kernel_regularizer=L1_reg)(x)


    model = keras.Model(inputs=[model_1.input, model_2.input], outputs=x)
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=opt)
    return model

def get_pgDNN(input_shape_1, input_shape_2, output_length, depth, bn_depth, lr, n_layers_1=1, n_layers_2=2, l1_penalty=0, activation=lrelu):
    '''
    Fully connected nerual network with 2 inputs, one at the start and one at a bottleneck
    '''
    L1_reg = keras.regularizers.L1(l1=l1_penalty)

    inputs_1 = keras.Input(shape=input_shape_1)
    x = inputs_1

    for i in range(n_layers_1):
        x = layers.Dense(depth,activation=activation, kernel_regularizer=L1_reg)(x)
    x = layers.Dense(bn_depth,activation=activation, kernel_regularizer=L1_reg)(x)
    model_1 = keras.Model(inputs_1, x)

    inputs_2 = keras.Input(shape=input_shape_2)
    x = inputs_2
    model_2 = keras.Model(inputs_2, x)

    combined_input = layers.concatenate([model_1.output, model_2.output])
    x = combined_input 
    for i in range(n_layers_2):
        x = layers.Dense(depth,activation=activation, kernel_regularizer=L1_reg)(x)
    x = layers.Dense(output_length,kernel_regularizer=L1_reg)(x)


    model = keras.Model(inputs=[model_1.input, model_2.input], outputs=x)
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=opt)
    return model

def get_LSTM(input_shape, output_length, dense_layers, dense_depth, lstm_layers, lstm_depth, lr, dropout_level=0):
    model = keras.Sequential()
    
    # Add correct amount of LSTM layers
    if lstm_layers == 1:
        model.add(layers.LSTM(lstm_depth, activation=lrelu, input_shape=input_shape, return_sequences=False))
    if lstm_layers  > 1:
        model.add(layers.LSTM(lstm_depth, activation=lrelu, input_shape=input_shape, return_sequences=True))
        for i in range(lstm_layers -1):
            model.add(layers.LSTM(lstm_depth, activation=lrelu, return_sequences=True))
        model.add(layers.LSTM(lstm_depth, activation=lrelu, return_sequences=False))
    model.add(layers.Dropout(dropout_level))

    for i in range(dense_layers):
        model.add(layers.Dense(dense_depth, activation=lrelu))
    model.add(layers.Dense(output_length))

    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=opt)
    return model

def get_pgLSTM(input_shape_1, input_shape_2, output_length, dense_layers, dense_depth, bn_depth, lstm_layers, lstm_depth, lr, dropout_level=0):
    inputs_1 = keras.Input(shape=input_shape_1)
    x = inputs_1
    # Add correct amount of LSTM layers
    if lstm_layers == 1:
        x = layers.LSTM(lstm_depth, activation=lrelu, input_shape=input_shape_1, return_sequences=False)(x)
    if lstm_layers  > 1:
        for i in range(lstm_layers-1):
            x = layers.LSTM(lstm_depth, activation=lrelu, input_shape=input_shape_1, return_sequences=True)(x)
        x = layers.LSTM(lstm_depth, activation=lrelu, return_sequences=False)(x)
    x = layers.Dense(bn_depth, activation=lrelu)(x)
    model_1 = keras.Model(inputs_1, x)

    inputs_2 = keras.Input(shape=input_shape_2)
    x = inputs_2
    model_2 = keras.Model(inputs_2, x)

    combined_input = layers.concatenate([model_1.output, model_2.output])
    x = combined_input 
    for i in range(dense_layers):
        x = layers.Dense(dense_depth, activation=lrelu)(x)
    x = layers.Dense(output_length)(x)


    model = keras.Model(inputs=[model_1.input, model_2.input], outputs=x)
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=opt)
    return model


class DNN_solver:
    def __init__(self, disc, epochs=[5000,5000], patience=[20,20], min_epochs=[20,20], noise_level=0, **NNkwargs):
        self.name = 'DNN'
        self.disc = disc
        self.normalize =True#False
        self.epochs = epochs
        self.patience = patience
        self.min_epochs = [max(min_epochs[i], patience[i]+2) for i in [0,1]] # assert min_epochs > patience
        self.noise_level = noise_level

        self.model = get_DNN(input_shape=(self.disc.Nv,), output_length = len(self.disc.inner_ids2), **NNkwargs)
        self.NNkwargs = NNkwargs

    def resetmodel(self):
        self.model = get_DNN(input_shape=(self.disc.Nv,), output_length = len(self.disc.inner_ids2), **self.NNkwargs)

    def __prep_data(self, data, set_norm_params):
        data = merge_first_dims(data)
        np.random.shuffle(data)
        X = data[:,:self.disc.Nv]
        Y = data[:,self.disc.Nv:]
        if set_norm_params: # only training set, not val
            self.mean = np.mean(X) if self.normalize else 0
            self.var = np.var(X) if self.normalize else 1
            self.Y_mean = np.mean(Y) if self.normalize else 0
            self.Y_var = np.var(Y) if self.normalize else 1
        X = X-self.mean
        X = X/self.var**0.5
        Y = Y-self.Y_mean
        Y = Y/self.Y_var**0.5
        X[:,self.disc.inner_ids2] += np.random.rand(X.shape[0],len(self.disc.inner_ids2))*self.noise_level-self.noise_level/2
        return X,Y

    def train(self, train_data, val_data, weights_loaded=False):
        X,Y = self.__prep_data(train_data['pnn'], True)
        if weights_loaded:
            return
        X_val,Y_val = self.__prep_data(val_data['pnn'], False)

        train_kwargs = {
                'batch_size':32,
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }

        train_kwargs = {
                'batch_size':32,
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }

        # train for minimum 100 steps
        train_result = self.model.fit(X, Y, epochs=self.min_epochs[1]-self.patience[1], **train_kwargs)
        self.train_hist = train_result.history['loss']
        self.val_hist = train_result.history['val_loss']

        # train with patience 20
        train_result = self.model.fit(X, Y,
                epochs=self.epochs[1],
                callbacks = [keras.callbacks.EarlyStopping(patience=self.patience[1])],
                **train_kwargs)
        self.train_hist.extend(train_result.history['loss'])
        self.val_hist.extend(train_result.history['val_loss'])


    def __call__(self, f, u_ex, alpha, callback=None, w_ex=None):
        plot_steps=False
        X=np.zeros((1,self.disc.Nv))
        u_NN = self.disc.format_u(u_ex(self.disc.pts[self.disc.inner_ids1], t=0))
        k=self.disc.k
        if plot_steps:
            plt.close()
        for t in range(self.disc.time_steps):
            u_prev = u_NN # save previous solution
            X[:,self.disc.inner_ids2] = u_prev
            X[:,self.disc.edge_ids2]=self.disc.format_u(u_ex(x=self.disc.pts[self.disc.edge_ids1],t=t*k))
            if plot_steps:
                if t%10==4:
                    print(f'time is {t*k}')
                    plt.plot(self.disc.pts, X[0,:self.disc.Nv],'r', label='new')
                    plt.plot(self.disc.pts, u_ex(x=self.disc.pts, t=t*k),'k--', label='ex')
            X = (X-self.mean)/self.var**0.5 # normalize NN input
            u_NN = self.model(X)
            u_NN = u_NN *self.Y_var**0.5 # unnormalize NN output
            u_NN = u_NN + self.Y_mean
            if callback!=None:
                callback((t+1)*k,u_NN)
        if plot_steps:
            plt.grid()
            #plt.legend()
            plt.show()
        result = np.zeros((self.disc.Nv))
        result[self.disc.inner_ids2] = u_NN
        result[self.disc.edge_ids2] = self.disc.format_u(u_ex(x=self.disc.pts[self.disc.edge_ids1],t=self.disc.T))
        return result

class pgLR_solver:
    def __init__(self, disc, epochs=[5000,5000], patience=[20,20], min_epochs=[20,20], noise_level=0, **NNkwargs):
        self.name = 'pgLR'
        self.disc = disc
        self.normalize =True#False
        self.epochs = epochs
        self.patience = patience
        self.min_epochs = [max(min_epochs[i], patience[i]+2) for i in [0,1]]
        self.noise_level = noise_level

        self.model = get_pgLR(input_shape_1=(self.disc.Nv,), input_shape_2=(2,), output_length = len(self.disc.inner_ids2), **NNkwargs)
        self.NNkwargs = NNkwargs

    def resetmodel(self):
        self.model = get_pgLR(input_shape_1=(self.disc.Nv,), input_shape_2=(2,), output_length = len(self.disc.inner_ids2), **self.NNkwargs)

    def __prep_data(self, data_1, data_2, set_norm_params):
        data_1 = merge_first_dims(data_1)
        data_2 = merge_first_dims(data_2)
        P = np.random.permutation(data_1.shape[0])
        data_1 = data_1[P,:]
        X1 = data_1[:,:self.disc.Nv]
        Y = data_1[:,self.disc.Nv:]
        X2 = data_2[P,:]
        if set_norm_params: # only training set, not val
            self.mean = np.mean(X1) if self.normalize else 0
            self.var = np.var(X1) if self.normalize else 1
            self.Y_mean = np.mean(Y) if self.normalize else 0
            self.Y_var = np.var(Y) if self.normalize else 1
            self.X2_mean = np.mean(X2) if self.normalize else 0
            self.X2_var = np.var(X2) if self.normalize else 1
        X1 = X1-self.mean
        X1 = X1/self.var**0.5
        Y = Y-self.Y_mean
        Y = Y/self.Y_var**0.5
        X2 = X2-self.X2_mean
        X2 = X2/self.X2_var**0.5
        X1[:,self.disc.inner_ids2] += np.random.rand(X1.shape[0],len(self.disc.inner_ids2))*self.noise_level-self.noise_level/2
        X = [X1,X2]
        return X,Y

    def train(self, train_data, val_data, weights_loaded=False):
        X,Y = self.__prep_data(train_data['pnn'], train_data['extra_feats'], True)
        if weights_loaded:
            #print(self.name)
            #test_linearity(self.model, input_shape_1=self.disc.Nv, input_shape_2=2, output_length = len(self.disc.inner_ids2))
            return
        X_val,Y_val = self.__prep_data(val_data['pnn'], val_data['extra_feats'], False)

        train_kwargs = {
                'batch_size':32,
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }

        train_kwargs = {
                'batch_size':32,
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }

        # train for minimum 100 steps
        train_result = self.model.fit(X, Y, epochs=self.min_epochs[1]-self.patience[1], **train_kwargs)
        self.train_hist = train_result.history['loss']
        self.val_hist = train_result.history['val_loss']

        # train with patience 20
        train_result = self.model.fit(X, Y,
                epochs=self.epochs[1],
                callbacks = [keras.callbacks.EarlyStopping(patience=self.patience[1])],
                **train_kwargs)
        self.train_hist.extend(train_result.history['loss'])
        self.val_hist.extend(train_result.history['val_loss'])

        #test_linearity(self.model, input_shape_1=self.disc.Nv, input_shape_2=2, output_length = len(self.disc.inner_ids2))

    def __call__(self, f, u_ex, alpha, callback=None, w_ex=None):
        X1=np.zeros((self.disc.Nv))
        u_NN = self.disc.format_u(u_ex(self.disc.pts[self.disc.inner_ids1], t=0))
        k=self.disc.k
        for t in range(self.disc.time_steps):
            u_prev = u_NN # save previous solution
            X1[self.disc.inner_ids2] = u_prev
            X1[self.disc.edge_ids2]=self.disc.format_u(u_ex(x=self.disc.pts[self.disc.edge_ids1],t=t*k))
            X2 = np.array([alpha,(t+1)*k])
            X1 = (X1-self.mean)/self.var**0.5 # normalize NN input
            X2 = (X2-self.X2_mean)/self.X2_var**0.5 # normalize NN input
            model_input = [np.array([X1]), np.array([X2])]
            u_NN = self.model(model_input)
            u_NN = u_NN *self.Y_var**0.5 # unnormalize NN output
            u_NN = u_NN + self.Y_mean
            if callback!=None:
                callback((t+1)*k,u_NN)
        result = np.zeros((self.disc.Nv))
        result[self.disc.inner_ids2] = u_NN
        result[self.disc.edge_ids2] = self.disc.format_u(u_ex(x=self.disc.pts[self.disc.edge_ids1],t=self.disc.T))
        return result

class pgDNN_solver:
    def __init__(self, disc, epochs=[5000,5000], patience=[20,20], min_epochs=[20,20], noise_level=0, **NNkwargs):
        self.name = 'pgDNN'
        self.disc = disc
        self.normalize =True#False
        self.epochs = epochs
        self.patience = patience
        self.min_epochs = [max(min_epochs[i], patience[i]+2) for i in [0,1]]
        self.noise_level = noise_level

        self.model = get_pgDNN(input_shape_1=(self.disc.Nv,), input_shape_2=(2,), output_length = len(self.disc.inner_ids2), **NNkwargs)
        self.NNkwargs = NNkwargs

    def resetmodel(self):
        self.model = get_pgDNN(input_shape_1=(self.disc.Nv,), input_shape_2=(2,), output_length = len(self.disc.inner_ids2), **self.NNkwargs)

    def __prep_data(self, data_1, data_2, set_norm_params):
        data_1 = merge_first_dims(data_1)
        data_2 = merge_first_dims(data_2)
        P = np.random.permutation(data_1.shape[0])
        data_1 = data_1[P,:]
        X1 = data_1[:,:self.disc.Nv]
        Y = data_1[:,self.disc.Nv:]
        X2 = data_2[P,:]
        if set_norm_params: # only training set, not val
            self.mean = np.mean(X1) if self.normalize else 0
            self.var = np.var(X1) if self.normalize else 1
            self.Y_mean = np.mean(Y) if self.normalize else 0
            self.Y_var = np.var(Y) if self.normalize else 1
            self.X2_mean = np.mean(X2) if self.normalize else 0
            self.X2_var = np.var(X2) if self.normalize else 1
        X1 = X1-self.mean
        X1 = X1/self.var**0.5
        Y = Y-self.Y_mean
        Y = Y/self.Y_var**0.5
        X2 = X2-self.X2_mean
        X2 = X2/self.X2_var**0.5
        X1[:,self.disc.inner_ids2] += np.random.rand(X1.shape[0],len(self.disc.inner_ids2))*self.noise_level-self.noise_level/2
        X = [X1,X2]
        return X,Y

    def train(self, train_data, val_data, weights_loaded=False):
        X,Y = self.__prep_data(train_data['pnn'], train_data['extra_feats'], True)
        if weights_loaded:
            #print(self.name)
            #test_linearity(self.model, input_shape_1=self.disc.Nv, input_shape_2=2, output_length = len(self.disc.inner_ids2))
            return
        X_val,Y_val = self.__prep_data(val_data['pnn'], val_data['extra_feats'], False)

        train_kwargs = {
                'batch_size':32,
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }

        train_kwargs = {
                'batch_size':32,
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }

        # train for minimum 100 steps
        train_result = self.model.fit(X, Y, epochs=self.min_epochs[1]-self.patience[1], **train_kwargs)
        self.train_hist = train_result.history['loss']
        self.val_hist = train_result.history['val_loss']

        # train with patience 20
        train_result = self.model.fit(X, Y,
                epochs=self.epochs[1],
                callbacks = [keras.callbacks.EarlyStopping(patience=self.patience[1])],
                **train_kwargs)
        self.train_hist.extend(train_result.history['loss'])
        self.val_hist.extend(train_result.history['val_loss'])
        #test_linearity(self.model, input_shape_1=self.disc.Nv, input_shape_2=2, output_length = len(self.disc.inner_ids2))


    def __call__(self, f, u_ex, alpha, callback=None, w_ex=None):
        X1=np.zeros((self.disc.Nv))
        u_NN = self.disc.format_u(u_ex(self.disc.pts[self.disc.inner_ids1], t=0))
        k=self.disc.k
        for t in range(self.disc.time_steps):
            u_prev = u_NN # save previous solution
            X1[self.disc.inner_ids2] = u_prev
            X1[self.disc.edge_ids2]=self.disc.format_u(u_ex(x=self.disc.pts[self.disc.edge_ids1],t=t*k))
            X2 = np.array([alpha,(t+1)*k])
            X1 = (X1-self.mean)/self.var**0.5 # normalize NN input
            X2 = (X2-self.X2_mean)/self.X2_var**0.5 # normalize NN input
            model_input = [np.array([X1]), np.array([X2])]
            u_NN = self.model(model_input)
            u_NN = u_NN *self.Y_var**0.5 # unnormalize NN output
            u_NN = u_NN + self.Y_mean
            if callback!=None:
                callback((t+1)*k,u_NN)
        result = np.zeros((self.disc.Nv))
        result[self.disc.inner_ids2] = u_NN
        result[self.disc.edge_ids2] = self.disc.format_u(u_ex(x=self.disc.pts[self.disc.edge_ids1],t=self.disc.T))
        return result


class LSTM_solver:
    def __init__(self, disc, epochs=[5000,5000], patience=[20,20], min_epochs=[20,20], noise_level=0, input_period=10, **NNkwargs):
        self.name = 'LSTM'
        self.disc = disc
        self.normalize =True#False
        self.epochs = epochs
        self.patience = patience
        self.min_epochs = [max(min_epochs[i], patience[i]+2) for i in [0,1]] # assert min_epochs > patience
        self.noise_level = noise_level
        self.input_period = input_period

        self.model = get_LSTM(input_shape=(self.input_period, self.disc.Nv,), output_length = len(self.disc.inner_ids2), **NNkwargs)
        self.NNkwargs = NNkwargs

    def resetmodel(self):
        self.model = get_LSTM(input_shape=(self.input_period, self.disc.Nv,), output_length = len(self.disc.inner_ids2), **self.NNkwargs)

    def __prep_data(self, data, set_norm_params):

        # Make X array with values from previous times
        l = self.input_period
        X = np.zeros((data.shape[0]+1-l, data.shape[1], l, self.disc.Nv))
        for t in range(1, 1+l):
            X[:,:,l-t,:] = data[l-t:data.shape[0]-t+1, :, :self.disc.Nv]
        X = merge_first_dims(X)
        Y = data[l-1:,:, self.disc.Nv:]
        Y = merge_first_dims(Y)

        # Shuffle input and output
        P = np.random.permutation(X.shape[0])
        X = X[P,:,:]
        Y = Y[P,:]

        if set_norm_params: # only training set, not val
            self.mean = np.mean(X) if self.normalize else 0
            self.var = np.var(X) if self.normalize else 1
            self.Y_mean = np.mean(Y) if self.normalize else 0
            self.Y_var = np.var(Y) if self.normalize else 1
        X = X-self.mean
        X = X/self.var**0.5
        Y = Y-self.Y_mean
        Y = Y/self.Y_var**0.5
        X[:,:,self.disc.inner_ids2] += np.random.rand(X.shape[0],l,len(self.disc.inner_ids2))*self.noise_level-self.noise_level/2
        return X,Y

    def train(self, train_data, val_data, weights_loaded=False):
        X,Y = self.__prep_data(train_data['pnn'], True)
        if weights_loaded:
            return
        X_val,Y_val = self.__prep_data(val_data['pnn'], False)

        train_kwargs = {
                'batch_size':32,
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }

        train_kwargs = {
                'batch_size':32,
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }

        # train for minimum 100 steps
        train_result = self.model.fit(X, Y, epochs=self.min_epochs[1]-self.patience[1], **train_kwargs)
        self.train_hist = train_result.history['loss']
        self.val_hist = train_result.history['val_loss']

        # train with patience 20
        train_result = self.model.fit(X, Y,
                epochs=self.epochs[1],
                callbacks = [keras.callbacks.EarlyStopping(patience=self.patience[1])],
                **train_kwargs)
        self.train_hist.extend(train_result.history['loss'])
        self.val_hist.extend(train_result.history['val_loss'])


    def __call__(self, f, u_ex, alpha, callback=None, w_ex=None):
        l=self.input_period
        X=np.zeros((1,l, self.disc.Nv))
        k=self.disc.k
        for t in range(1,l+1):
            X[:,l-t,:] = self.disc.format_u(u_ex(self.disc.pts, t=-t*k)) # TODO: find a porper way to do this (backward exxtrapolation not always possible
        X = (X-self.mean)/self.var**0.5 # normalize NN input (only last entry)
        u_NN = self.disc.format_u(u_ex(self.disc.pts[self.disc.inner_ids1], t=0))
        for t in range(self.disc.time_steps):
            X[:,:-1,:] = X[:,1:,:] # move input data one step forward
            u_prev = u_NN # save previous solution
            X[:,-1:,self.disc.inner_ids2] = u_prev
            X[:,-1,self.disc.edge_ids2]=self.disc.format_u(u_ex(x=self.disc.pts[self.disc.edge_ids1],t=t*k))
            X[:,-1,:] = (X[:,-1,:]-self.mean)/self.var**0.5 # normalize NN input (only last entry)
            u_NN = self.model(X)
            u_NN = u_NN *self.Y_var**0.5 # unnormalize NN output
            u_NN = u_NN + self.Y_mean
            if callback!=None:
                callback((t+1)*k,u_NN)
        result = np.zeros((self.disc.Nv))
        result[self.disc.inner_ids2] = u_NN
        result[self.disc.edge_ids2] = self.disc.format_u(u_ex(x=self.disc.pts[self.disc.edge_ids1],t=self.disc.T))
        return result

class pgLSTM_solver:
    def __init__(self, disc, epochs=[5000,5000], patience=[20,20], min_epochs=[20,20], noise_level=0, input_period=10, **NNkwargs):
        self.name = 'pgLSTM'
        self.disc = disc
        self.normalize =True#False
        self.epochs = epochs
        self.patience = patience
        self.min_epochs = [max(min_epochs[i], patience[i]+2) for i in [0,1]] # assert min_epochs > patience
        self.noise_level = noise_level
        self.input_period = input_period

        self.model = get_pgLSTM(input_shape_1=(self.input_period, self.disc.Nv,), input_shape_2=(2,), output_length = len(self.disc.inner_ids2), **NNkwargs)
        self.NNkwargs = NNkwargs

    def resetmodel(self):
        self.model = get_pgLSTM(input_shape_1=(self.input_period, self.disc.Nv,), input_shape_2=(2,), output_length = len(self.disc.inner_ids2), **self.NNkwargs)

    def __prep_data(self, data_1, data_2, set_norm_params):

        # Make X array with values from previous times
        l = self.input_period
        X1 = np.zeros((data_1.shape[0]+1-l, data_1.shape[1], l, self.disc.Nv))
        for t in range(1, 1+l):
            X1[:,:,l-t,:] = data_1[l-t:data_1.shape[0]-t+1, :, :self.disc.Nv]
        X1 = merge_first_dims(X1)
        X2 = merge_first_dims(data_2)
        Y = data_1[l-1:,:, self.disc.Nv:]
        Y = merge_first_dims(Y)

        # Shuffle input and output
        P = np.random.permutation(X1.shape[0])
        X1 = X1[P,:,:]
        X2 = X2[P,:]
        Y = Y[P,:]

        if set_norm_params: # only training set, not val
            self.mean = np.mean(X1) if self.normalize else 0
            self.var = np.var(X1) if self.normalize else 1
            self.Y_mean = np.mean(Y) if self.normalize else 0
            self.Y_var = np.var(Y) if self.normalize else 1
            self.X2_mean = np.mean(X2) if self.normalize else 0
            self.X2_var = np.var(X2) if self.normalize else 1
        X1 = X1-self.mean
        X1 = X1/self.var**0.5
        Y = Y-self.Y_mean
        Y = Y/self.Y_var**0.5
        X2 = X2-self.X2_mean
        X2 = X2/self.X2_var**0.5
        X1[:,:,self.disc.inner_ids2] += np.random.rand(X1.shape[0],l,len(self.disc.inner_ids2))*self.noise_level-self.noise_level/2
        X = [X1,X2]
        return X,Y

    def train(self, train_data, val_data, weights_loaded=False):
        X,Y = self.__prep_data(train_data['pnn'], train_data['extra_feats'], True)
        if weights_loaded:
            return
        X_val,Y_val = self.__prep_data(val_data['pnn'], val_data['extra_feats'], False)

        train_kwargs = {
                'batch_size':32,
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }

        train_kwargs = {
                'batch_size':32,
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }

        # train for minimum 100 steps
        train_result = self.model.fit(X, Y, epochs=self.min_epochs[1]-self.patience[1], **train_kwargs)
        self.train_hist = train_result.history['loss']
        self.val_hist = train_result.history['val_loss']

        # train with patience 20
        train_result = self.model.fit(X, Y,
                epochs=self.epochs[1],
                callbacks = [keras.callbacks.EarlyStopping(patience=self.patience[1])],
                **train_kwargs)
        self.train_hist.extend(train_result.history['loss'])
        self.val_hist.extend(train_result.history['val_loss'])


    def __call__(self, f, u_ex, alpha, callback=None, w_ex=None):
        l=self.input_period
        X1=np.zeros((1,l, self.disc.Nv))
        k=self.disc.k
        for t in range(1,l+1):
            X1[:,l-t,:] = self.disc.format_u(u_ex(self.disc.pts, t=-t*k)) # TODO: find a porper way to do this (backward exxtrapolation not always possible
        X1 = (X1-self.mean)/self.var**0.5 # normalize NN input (only last entry)
        u_NN = self.disc.format_u(u_ex(self.disc.pts[self.disc.inner_ids1], t=0))
        for t in range(self.disc.time_steps):
            X1[:,:-1,:] = X1[:,1:,:] # move input data one step forward
            u_prev = u_NN # save previous solution
            X1[:,-1:,self.disc.inner_ids2] = u_prev
            X1[:,-1,self.disc.edge_ids2]=self.disc.format_u(u_ex(x=self.disc.pts[self.disc.edge_ids1],t=t*k))
            X1[:,-1,:] = (X1[:,-1,:]-self.mean)/self.var**0.5 # normalize NN input (only last entry)
            X2 = np.array([alpha,(t+1)*k])
            X2 = (X2-self.X2_mean)/self.X2_var**0.5 # normalize NN input
            model_input = [np.array(X1), np.array([X2])]
            u_NN = self.model(model_input)
            u_NN = u_NN *self.Y_var**0.5 # unnormalize NN output
            u_NN = u_NN + self.Y_mean
            if callback!=None:
                callback((t+1)*k,u_NN)
        result = np.zeros((self.disc.Nv))
        result[self.disc.inner_ids2] = u_NN
        result[self.disc.edge_ids2] = self.disc.format_u(u_ex(x=self.disc.pts[self.disc.edge_ids1],t=self.disc.T))
        return result


class CoSTA_DNN_solver:
    def __init__(self, disc, epochs=[5000,5000], patience=[20,20], min_epochs=[20,20], noise_level=0, **NNkwargs):
        self.name = 'CoSTA_DNN'
        self.disc = disc
        self.normalize =True#False
        self.epochs = epochs
        self.patience = patience
        self.min_epochs = [max(min_epochs[i], patience[i]+2) for i in [0,1]] # assert min_epochs > patience
        self.noise_level = noise_level

        self.model = get_DNN(input_shape=(self.disc.Nv,), output_length = len(self.disc.inner_ids2), **NNkwargs)
        self.NNkwargs = NNkwargs

    def resetmodel(self):
        self.model = get_DNN(input_shape=(self.disc.Nv,), output_length = len(self.disc.inner_ids2), **self.NNkwargs)

    def __prep_data(self, data, set_norm_params):
        data = merge_first_dims(data)
        np.random.shuffle(data)
        X = data[:,:self.disc.Nv]
        Y = data[:,self.disc.Nv:]
        if set_norm_params: # only training set, not val
            self.mean = np.mean(X) if self.normalize else 0
            self.var = np.var(X) if self.normalize else 1
            self.Y_mean = np.mean(Y) if self.normalize else 0
            self.Y_var = np.var(Y) if self.normalize else 1
        X = X-self.mean
        X = X/self.var**0.5
        Y = Y-self.Y_mean
        Y = Y/self.Y_var**0.5
        X[:,self.disc.inner_ids2] += np.random.rand(X.shape[0],len(self.disc.inner_ids2))*self.noise_level-self.noise_level/2
        return X,Y

    def train(self, train_data, val_data, weights_loaded=False):
        X,Y = self.__prep_data(train_data['ham'], True)
        if weights_loaded:
            return
        X_val,Y_val = self.__prep_data(val_data['ham'], False)

        train_kwargs = {
                'batch_size':32,
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }

        train_kwargs = {
                'batch_size':32,
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }

        # train for minimum 100 steps
        train_result = self.model.fit(X, Y, epochs=self.min_epochs[0]-self.patience[0], **train_kwargs)
        self.train_hist = train_result.history['loss']
        self.val_hist = train_result.history['val_loss']

        # train with patience 20
        train_result = self.model.fit(X, Y,
                epochs=self.epochs[0],
                callbacks = [keras.callbacks.EarlyStopping(patience=self.patience[0])],
                **train_kwargs)
        self.train_hist.extend(train_result.history['loss'])
        self.val_hist.extend(train_result.history['val_loss'])

    def __call__(self, f, u_ex, alpha, callback=None, w_ex=None):
        X=np.zeros((1,self.disc.Nv))

        fem_model = self.disc.make_model(f, u_ex, w_ex=w_ex)
        fem_model.u_fem =self.disc.format_u(fem_model.u_ex(fem_model.pts, t=fem_model.time))
        fem_model.w_fem = None # to avoid many if statements later
        if self.disc.equation == 'elasticity':
            fem_model.w_fem =self.disc.format_u(fem_model.w_ex(fem_model.pts, t=fem_model.time))
        for t in range(self.disc.time_steps):
            u_prev = fem_model.u_fem # save previous solution
            w_prev = fem_model.w_fem # save previous solution
            fem_model.step() # first, uncorrected, step
            fem_step = fem_model.u_fem
            X[:,:self.disc.Nv] = fem_step
            X = (X-self.mean)/self.var**0.5
            correction = np.zeros(self.disc.Nv)
            correction[self.disc.inner_ids2] = self.model(X)[0,:]
            correction[self.disc.inner_ids2] = correction[self.disc.inner_ids2]*self.Y_var**0.5
            correction[self.disc.inner_ids2] = correction[self.disc.inner_ids2] + self.Y_mean
            fem_model.u_fem = u_prev
            fem_model.w_fem = w_prev
            fem_model.time -= fem_model.k # set back time for correction
            fem_model.step(correction=correction) # corrected step
            if callback!=None:
                callback(fem_model.time, fem_model.u_fem)
        return fem_model.u_fem


class CoSTA_pgLR_solver:
    def __init__(self, disc, epochs=[5000,5000], patience=[20,20], min_epochs=[20,20], noise_level=0, **NNkwargs):
        self.name = 'CoSTA_pgLR'
        self.disc = disc
        self.normalize =True#False
        self.epochs = epochs
        self.patience = patience
        self.min_epochs = [max(min_epochs[i], patience[i]+2) for i in [0,1]] # assert min_epochs > patience
        self.noise_level = noise_level

        self.model = get_pgLR(input_shape_1=(self.disc.Nv,), input_shape_2=(2,), output_length = len(self.disc.inner_ids2), **NNkwargs)
        self.NNkwargs = NNkwargs

    def resetmodel(self):
        self.model = get_pgLR(input_shape_1=(self.disc.Nv,), input_shape_2=(2,), output_length = len(self.disc.inner_ids2), **self.NNkwargs)

    def __prep_data(self, data_1, data_2, set_norm_params):
        data_1 = merge_first_dims(data_1)
        data_2 = merge_first_dims(data_2)
        P = np.random.permutation(data_1.shape[0])
        data_1 = data_1[P,:]
        X1 = data_1[:,:self.disc.Nv]
        Y = data_1[:,self.disc.Nv:]
        X2 = data_2[P,:]
        if set_norm_params: # only training set, not val
            self.mean = np.mean(X1) if self.normalize else 0
            self.var = np.var(X1) if self.normalize else 1
            self.Y_mean = np.mean(Y) if self.normalize else 0
            self.Y_var = np.var(Y) if self.normalize else 1
            self.X2_mean = np.mean(X2) if self.normalize else 0
            self.X2_var = np.var(X2) if self.normalize else 1
        X1 = X1-self.mean
        X1 = X1/self.var**0.5
        Y = Y-self.Y_mean
        Y = Y/self.Y_var**0.5
        X2 = X2-self.X2_mean
        X2 = X2/self.X2_var**0.5
        X1[:,self.disc.inner_ids2] += np.random.rand(X1.shape[0],len(self.disc.inner_ids2))*self.noise_level-self.noise_level/2
        X = [X1,X2]
        return X,Y

    def train(self, train_data, val_data, weights_loaded=False):
        X,Y = self.__prep_data(train_data['ham'], train_data['extra_feats'], True)
        if weights_loaded:
            #print(self.name)
            #test_linearity(self.model, input_shape_1=self.disc.Nv, input_shape_2=2, output_length = len(self.disc.inner_ids2))
            return
        X_val,Y_val = self.__prep_data(val_data['ham'], val_data['extra_feats'], False)

        train_kwargs = {
                'batch_size':32,
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }

        train_kwargs = {
                'batch_size':32,
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }

        # train for minimum 100 steps
        train_result = self.model.fit(X, Y, epochs=self.min_epochs[0]-self.patience[0], **train_kwargs)
        self.train_hist = train_result.history['loss']
        self.val_hist = train_result.history['val_loss']

        # train with patience 20
        train_result = self.model.fit(X, Y,
                epochs=self.epochs[0],
                callbacks = [keras.callbacks.EarlyStopping(patience=self.patience[0])],
                **train_kwargs)
        self.train_hist.extend(train_result.history['loss'])
        self.val_hist.extend(train_result.history['val_loss'])
        
        #test_linearity(self.model, input_shape_1=self.disc.Nv, input_shape_2=2, output_length = len(self.disc.inner_ids2))


    def __call__(self, f, u_ex, alpha, callback=None, w_ex=None):
        X1=np.zeros((self.disc.Nv))

        fem_model = self.disc.make_model(f, u_ex, w_ex=w_ex)
        fem_model.u_fem =self.disc.format_u(fem_model.u_ex(fem_model.pts, t=fem_model.time))
        fem_model.w_fem = None # to avoid many if statements later
        if self.disc.equation == 'elasticity':
            fem_model.w_fem =self.disc.format_u(fem_model.w_ex(fem_model.pts, t=fem_model.time))
        for t in range(self.disc.time_steps):
            u_prev = fem_model.u_fem # save previous solution
            w_prev = fem_model.w_fem # save previous solution
            fem_model.step() # first, uncorrected, step
            fem_step = fem_model.u_fem
            X1[:self.disc.Nv] = fem_step
            X1 = (X1-self.mean)/self.var**0.5
            X2 = np.array([alpha,fem_model.time])
            X2 = (X2-self.X2_mean)/self.X2_var**0.5
            model_input = [np.array([X1]), np.array([X2])]
            correction = np.zeros(self.disc.Nv)
            correction[self.disc.inner_ids2] = self.model(model_input)[0,:]
            correction[self.disc.inner_ids2] = correction[self.disc.inner_ids2]*self.Y_var**0.5
            correction[self.disc.inner_ids2] = correction[self.disc.inner_ids2] + self.Y_mean
            fem_model.u_fem = u_prev
            fem_model.w_fem = w_prev
            fem_model.time -= fem_model.k # set back time for correction
            fem_model.step(correction=correction) # corrected step
            if callback!=None:
                callback(fem_model.time, fem_model.u_fem)
        return fem_model.u_fem


class CoSTA_pgDNN_solver:
    def __init__(self, disc, epochs=[5000,5000], patience=[20,20], min_epochs=[20,20], noise_level=0, **NNkwargs):
        self.name = 'CoSTA_pgDNN'
        self.disc = disc
        self.normalize =True#False
        self.epochs = epochs
        self.patience = patience
        self.min_epochs = [max(min_epochs[i], patience[i]+2) for i in [0,1]] # assert min_epochs > patience
        self.noise_level = noise_level

        self.model = get_pgDNN(input_shape_1=(self.disc.Nv,), input_shape_2=(2,), output_length = len(self.disc.inner_ids2), **NNkwargs)
        self.NNkwargs = NNkwargs

    def resetmodel(self):
        self.model = get_pgDNN(input_shape_1=(self.disc.Nv,), input_shape_2=(2,), output_length = len(self.disc.inner_ids2), **self.NNkwargs)

    def __prep_data(self, data_1, data_2, set_norm_params):
        data_1 = merge_first_dims(data_1)
        data_2 = merge_first_dims(data_2)
        P = np.random.permutation(data_1.shape[0])
        data_1 = data_1[P,:]
        X1 = data_1[:,:self.disc.Nv]
        Y = data_1[:,self.disc.Nv:]
        X2 = data_2[P,:]
        if set_norm_params: # only training set, not val
            self.mean = np.mean(X1) if self.normalize else 0
            self.var = np.var(X1) if self.normalize else 1
            self.Y_mean = np.mean(Y) if self.normalize else 0
            self.Y_var = np.var(Y) if self.normalize else 1
            self.X2_mean = np.mean(X2) if self.normalize else 0
            self.X2_var = np.var(X2) if self.normalize else 1
        X1 = X1-self.mean
        X1 = X1/self.var**0.5
        Y = Y-self.Y_mean
        Y = Y/self.Y_var**0.5
        X2 = X2-self.X2_mean
        X2 = X2/self.X2_var**0.5
        X1[:,self.disc.inner_ids2] += np.random.rand(X1.shape[0],len(self.disc.inner_ids2))*self.noise_level-self.noise_level/2
        X = [X1,X2]
        return X,Y

    def train(self, train_data, val_data, weights_loaded=False):
        X,Y = self.__prep_data(train_data['ham'], train_data['extra_feats'], True)
        if weights_loaded:
            #print(self.name)
            #test_linearity(self.model, input_shape_1=self.disc.Nv, input_shape_2=2, output_length = len(self.disc.inner_ids2))
            return
        X_val,Y_val = self.__prep_data(val_data['ham'], val_data['extra_feats'], False)

        train_kwargs = {
                'batch_size':32,
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }

        train_kwargs = {
                'batch_size':32,
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }

        # train for minimum 100 steps
        train_result = self.model.fit(X, Y, epochs=self.min_epochs[0]-self.patience[0], **train_kwargs)
        self.train_hist = train_result.history['loss']
        self.val_hist = train_result.history['val_loss']

        # train with patience 20
        train_result = self.model.fit(X, Y,
                epochs=self.epochs[0],
                callbacks = [keras.callbacks.EarlyStopping(patience=self.patience[0])],
                **train_kwargs)
        self.train_hist.extend(train_result.history['loss'])
        self.val_hist.extend(train_result.history['val_loss'])
        #test_linearity(self.model, input_shape_1=self.disc.Nv, input_shape_2=2, output_length = len(self.disc.inner_ids2))


    def __call__(self, f, u_ex, alpha, callback=None, w_ex=None):
        X1=np.zeros((self.disc.Nv))

        fem_model = self.disc.make_model(f, u_ex, w_ex=w_ex)
        fem_model.u_fem =self.disc.format_u(fem_model.u_ex(fem_model.pts, t=fem_model.time))
        fem_model.w_fem = None # to avoid many if statements later
        if self.disc.equation == 'elasticity':
            fem_model.w_fem =self.disc.format_u(fem_model.w_ex(fem_model.pts, t=fem_model.time))
        for t in range(self.disc.time_steps):
            u_prev = fem_model.u_fem # save previous solution
            w_prev = fem_model.w_fem # save previous solution
            fem_model.step() # first, uncorrected, step
            fem_step = fem_model.u_fem
            X1[:self.disc.Nv] = fem_step
            X1 = (X1-self.mean)/self.var**0.5
            X2 = np.array([alpha,fem_model.time])
            X2 = (X2-self.X2_mean)/self.X2_var**0.5
            model_input = [np.array([X1]), np.array([X2])]
            correction = np.zeros(self.disc.Nv)
            correction[self.disc.inner_ids2] = self.model(model_input)[0,:]
            correction[self.disc.inner_ids2] = correction[self.disc.inner_ids2]*self.Y_var**0.5
            correction[self.disc.inner_ids2] = correction[self.disc.inner_ids2] + self.Y_mean
            fem_model.u_fem = u_prev
            fem_model.w_fem = w_prev
            fem_model.time -= fem_model.k # set back time for correction
            fem_model.step(correction=correction) # corrected step
            if callback!=None:
                callback(fem_model.time, fem_model.u_fem)
        return fem_model.u_fem


class CoSTA_LSTM_solver:
    def __init__(self, disc, epochs=[5000,5000], patience=[20,20], min_epochs=[20,20], noise_level=0, input_period=10, **NNkwargs):
        self.name = 'CoSTA_LSTM'
        self.disc = disc
        self.normalize =True#False
        self.epochs = epochs
        self.patience = patience
        self.min_epochs = [max(min_epochs[i], patience[i]+2) for i in [0,1]] # assert min_epochs > patience
        self.noise_level = noise_level
        self.input_period = input_period

        self.model = get_LSTM(input_shape=(self.input_period, self.disc.Nv,), output_length = len(self.disc.inner_ids2), **NNkwargs)
        self.NNkwargs = NNkwargs

    def resetmodel(self):
        self.model = get_LSTM(input_shape=(self.input_period, self.disc.Nv,), output_length = len(self.disc.inner_ids2), **self.NNkwargs)

    def __prep_data(self, data, set_norm_params):

        # Make X array with values from previous times
        l = self.input_period
        X = np.zeros((data.shape[0]+1-l, data.shape[1], l, self.disc.Nv))
        for t in range(1, 1+l):
            X[:,:,l-t,:] = data[l-t:data.shape[0]-t+1, :, :self.disc.Nv]
        X = merge_first_dims(X)
        Y = data[l-1:,:, self.disc.Nv:]
        Y = merge_first_dims(Y)

        # Shuffle input and output
        P = np.random.permutation(X.shape[0])
        X = X[P,:,:]
        Y = Y[P,:]

        if set_norm_params: # only training set, not val
            self.mean = np.mean(X) if self.normalize else 0
            self.var = np.var(X) if self.normalize else 1
            self.Y_mean = np.mean(Y) if self.normalize else 0
            self.Y_var = np.var(Y) if self.normalize else 1
        X = X-self.mean
        X = X/self.var**0.5
        Y = Y-self.Y_mean
        Y = Y/self.Y_var**0.5
        X[:,:,self.disc.inner_ids2] += np.random.rand(X.shape[0],l,len(self.disc.inner_ids2))*self.noise_level-self.noise_level/2
        return X,Y

    def train(self, train_data, val_data, weights_loaded=False):
        X,Y = self.__prep_data(train_data['ham'], True)
        if weights_loaded:
            return
        X_val,Y_val = self.__prep_data(val_data['ham'], False)

        train_kwargs = {
                'batch_size':32,
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }

        train_kwargs = {
                'batch_size':32,
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }

        # train for minimum 100 steps
        train_result = self.model.fit(X, Y, epochs=self.min_epochs[0]-self.patience[0], **train_kwargs)
        self.train_hist = train_result.history['loss']
        self.val_hist = train_result.history['val_loss']

        # train with patience 20
        train_result = self.model.fit(X, Y,
                epochs=self.epochs[0],
                callbacks = [keras.callbacks.EarlyStopping(patience=self.patience[0])],
                **train_kwargs)
        self.train_hist.extend(train_result.history['loss'])
        self.val_hist.extend(train_result.history['val_loss'])

    def __call__(self, f, u_ex, alpha, callback=None, w_ex=None):
        l=self.input_period
        X=np.zeros((1,l, self.disc.Nv))
        k=self.disc.k
        X[:,0,:] = self.disc.format_u(u_ex(self.disc.pts, t=0))

        # Define fem model
        fem_model = self.disc.make_model(f, u_ex, w_ex=w_ex)

        # Calculate initial input vector
        for t in range(1,l+1):
            fem_model.u_fem = self.disc.format_u(u_ex(self.disc.pts, t=-t*k)) # TODO: find a porper way to do this (backward exxtrapolation not always possible
            if self.disc.equation == 'elasticity':
                fem_model.w_fem =self.disc.format_u(fem_model.w_ex(self.disc.pts, t=-t*k))
            fem_model.time = -t*k
            fem_model.step() # make one step from exact solution (as the LSTM model is trained on)
            X[:,l-t,:] = fem_model.u_fem
        X = (X-self.mean)/self.var**0.5 # normalize NN input
  
        # Reset fem model
        fem_model.time = 0
        fem_model.u_fem =self.disc.format_u(fem_model.u_ex(fem_model.pts, t=fem_model.time))
        fem_model.w_fem = None # to avoid many if statements later
        if self.disc.equation == 'elasticity':
            fem_model.w_fem =self.disc.format_u(fem_model.w_ex(fem_model.pts, t=fem_model.time))

        for t in range(self.disc.time_steps):
            X[:,:-1,:] = X[:,1:,:] # move input data one step forward
            u_prev = fem_model.u_fem # save previous solution
            w_prev = fem_model.w_fem # save previous solution
            fem_model.step() # first, uncorrected, step
            fem_step = fem_model.u_fem
            X[:,-1:,:self.disc.Nv] = fem_step
            X[:,-1,:] = (X[:,-1,:]-self.mean)/self.var**0.5 # normalize NN input (only last entry)
            correction = np.zeros(self.disc.Nv)
            correction[self.disc.inner_ids2] = self.model(X)[0,:]
            correction[self.disc.inner_ids2] = correction[self.disc.inner_ids2]*self.Y_var**0.5
            correction[self.disc.inner_ids2] = correction[self.disc.inner_ids2] + self.Y_mean
            fem_model.u_fem = u_prev
            fem_model.w_fem = w_prev
            fem_model.time -= fem_model.k # set back time for correction
            fem_model.step(correction=correction) # corrected step
            if callback!=None:
                callback(fem_model.time, fem_model.u_fem)
        return fem_model.u_fem

class CoSTA_pgLSTM_solver:
    def __init__(self, disc, epochs=[5000,5000], patience=[20,20], min_epochs=[20,20], noise_level=0, input_period=10, **NNkwargs):
        self.name = 'CoSTA_pgLSTM'
        self.disc = disc
        self.normalize =True#False
        self.epochs = epochs
        self.patience = patience
        self.min_epochs = [max(min_epochs[i], patience[i]+2) for i in [0,1]] # assert min_epochs > patience
        self.noise_level = noise_level
        self.input_period = input_period

        self.model = get_pgLSTM(input_shape_1=(self.input_period, self.disc.Nv,), input_shape_2=(2,), output_length = len(self.disc.inner_ids2), **NNkwargs)
        self.NNkwargs = NNkwargs

    def resetmodel(self):
        self.model = get_pgLSTM(input_shape_1=(self.input_period, self.disc.Nv,), input_shape_2=(2,), output_length = len(self.disc.inner_ids2), **self.NNkwargs)

    def __prep_data(self, data_1, data_2, set_norm_params):

        # Make X array with values from previous times
        l = self.input_period
        X1 = np.zeros((data_1.shape[0]+1-l, data_1.shape[1], l, self.disc.Nv))
        for t in range(1, 1+l):
            X1[:,:,l-t,:] = data_1[l-t:data_1.shape[0]-t+1, :, :self.disc.Nv]
        X1 = merge_first_dims(X1)
        X2 = merge_first_dims(data_2)
        Y = data_1[l-1:,:, self.disc.Nv:]
        Y = merge_first_dims(Y)

        # Shuffle input and output
        P = np.random.permutation(X1.shape[0])
        X1 = X1[P,:,:]
        X2 = X2[P,:]
        Y = Y[P,:]

        if set_norm_params: # only training set, not val
            self.mean = np.mean(X1) if self.normalize else 0
            self.var = np.var(X1) if self.normalize else 1
            self.Y_mean = np.mean(Y) if self.normalize else 0
            self.Y_var = np.var(Y) if self.normalize else 1
            self.X2_mean = np.mean(X2) if self.normalize else 0
            self.X2_var = np.var(X2) if self.normalize else 1
        X1 = X1-self.mean
        X1 = X1/self.var**0.5
        Y = Y-self.Y_mean
        Y = Y/self.Y_var**0.5
        X2 = X2-self.X2_mean
        X2 = X2/self.X2_var**0.5
        X1[:,:,self.disc.inner_ids2] += np.random.rand(X1.shape[0],l,len(self.disc.inner_ids2))*self.noise_level-self.noise_level/2
        X = [X1,X2]
        return X,Y

    def train(self, train_data, val_data, weights_loaded=False):
        X,Y = self.__prep_data(train_data['pnn'], train_data['extra_feats'], True)
        if weights_loaded:
            return
        X_val,Y_val = self.__prep_data(val_data['pnn'], val_data['extra_feats'], False)

        train_kwargs = {
                'batch_size':32,
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }

        train_kwargs = {
                'batch_size':32,
                'validation_data':(X_val, Y_val),
                'verbose':0,
                }

        # train for minimum 100 steps
        train_result = self.model.fit(X, Y, epochs=self.min_epochs[0]-self.patience[0], **train_kwargs)
        self.train_hist = train_result.history['loss']
        self.val_hist = train_result.history['val_loss']

        # train with patience 20
        train_result = self.model.fit(X, Y,
                epochs=self.epochs[0],
                callbacks = [keras.callbacks.EarlyStopping(patience=self.patience[0])],
                **train_kwargs)
        self.train_hist.extend(train_result.history['loss'])
        self.val_hist.extend(train_result.history['val_loss'])

    def __call__(self, f, u_ex, alpha, callback=None, w_ex=None):
        l=self.input_period
        X1=np.zeros((1,l, self.disc.Nv))
        k=self.disc.k
        X1[:,0,:] = self.disc.format_u(u_ex(self.disc.pts, t=0))

        # Define fem model
        fem_model = self.disc.make_model(f, u_ex, w_ex=w_ex)

        # Calculate initial input vector
        for t in range(1,l+1):
            fem_model.u_fem = self.disc.format_u(u_ex(self.disc.pts, t=-t*k)) # TODO: find a porper way to do this (backward exxtrapolation not always possible
            if self.disc.equation == 'elasticity':
                fem_model.w_fem =self.disc.format_u(fem_model.w_ex(self.disc.pts, t=-t*k))
            fem_model.time = -t*k
            fem_model.step() # make one step from exact solution (as the LSTM model is trained on)
            X1[:,l-t,:] = fem_model.u_fem
        X1 = (X1-self.mean)/self.var**0.5 # normalize NN input
  
        # Reset fem model
        fem_model.time = 0
        fem_model.u_fem =self.disc.format_u(fem_model.u_ex(fem_model.pts, t=fem_model.time))
        fem_model.w_fem = None # to avoid many if statements later
        if self.disc.equation == 'elasticity':
            fem_model.w_fem =self.disc.format_u(fem_model.w_ex(fem_model.pts, t=fem_model.time))

        for t in range(self.disc.time_steps):
            X1[:,:-1,:] = X1[:,1:,:] # move input data one step forward
            u_prev = fem_model.u_fem # save previous solution
            w_prev = fem_model.w_fem # save previous solution
            fem_model.step() # first, uncorrected, step
            fem_step = fem_model.u_fem
            X1[:,-1:,:self.disc.Nv] = fem_step
            X1[:,-1,:] = (X1[:,-1,:]-self.mean)/self.var**0.5 # normalize NN input (only last entry)
            X2 = np.array([alpha,fem_model.time])
            X2 = (X2-self.X2_mean)/self.X2_var**0.5
            model_input = [np.array(X1), np.array([X2])]
            correction = np.zeros(self.disc.Nv)
            correction[self.disc.inner_ids2] = self.model(model_input)[0,:]
            correction[self.disc.inner_ids2] = correction[self.disc.inner_ids2]*self.Y_var**0.5
            correction[self.disc.inner_ids2] = correction[self.disc.inner_ids2] + self.Y_mean
            fem_model.u_fem = u_prev
            fem_model.w_fem = w_prev
            fem_model.time -= fem_model.k # set back time for correction
            fem_model.step(correction=correction) # corrected step
            if callback!=None:
                callback(fem_model.time, fem_model.u_fem)
        return fem_model.u_fem
