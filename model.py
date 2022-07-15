import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks
import keras.optimizers
import keras.losses

class TripletLoss:
    # Basis for code pulled from "Deep triplet hashing network for case based medical image retrieval",
    #referenced as basis for loss function l1 in the paper.
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin  # margin threshold
        self.mse_loss = keras.losses.mean_squared_error(reduction='none')

    def forward(self, bq, bp, bn):
        margin_val = self.margin * bq.shape[1]
        squared_loss_pos = tf.reduce_mean(self.mse_loss(bq, bp), dim=1) #reduce_mean on positive and anchor
        squared_loss_neg = tf.reduce_mean(self.mse_loss(bq, bn), dim=1) #reduce_mean on negative and anchor
        zeros = tf.zeros_like(squared_loss_neg)
        loss = tf.math.reduce_max(zeros, margin_val - squared_loss_neg + squared_loss_pos)
        return tf.math.reduce_max(loss)

class Network:
    def __init__(self, shape=(67,1,64,128,128)):
        self.hash_bits = hash_bits
        self.model = self.triplets_model(shape, dimensions)
        self.compile = self.model.compile(optimizer = tf.keras.optimizers.adam_v2.Adam(lr=1e-4),
                           metrics=['accuracy'])
        # Need to properly incorporate loss into above
        self.fit = self.model.fit
        self.predict = self.model.predict
        self.evaluate = self.model.evaluate
        self.summary = self.model.summary
    def triplets_model(self, shape):
        # Takes the encoder and decoder and applies them to the triplet, which is formatted into an input. Calculates l1,l2,l3, and the final loss function L.
        # Integer values in L are the w-value hyperparameters from the study
        # Returns a model which takes 3 inputs and returns the loss.
        xq = Input(shape=shape, name='xq')
        xp = Input(shape=shape, name='xp')
        xn = Input(shape=shape, name='xn')

        embed = self.encoder(shape)
        bq = embed(xq)
        bp = embed(xp)
        bn = embed(xn)

        decode = self.decoder()
        yq = decode(bq)
        yp = decode(bp)
        yn = decode(bn)

        def norm(x):
            # Normalization function
            return sqrt(sum_all(square(x)))

        con_loss = TripletLoss(margin=0.5).cuda()
        l1 = con_loss(bq, bp, bn)
        l2 = pow(norm(abs(bq) - 1), 2) + pow(norm(abs(bp) - 1), 2) + pow(norm(abs(bn) - 1), 2)
        l3 = sum(pow(norm(xq - yq), 2) + pow(norm(xp - yp), 2) + pow(norm(xn - yn), 2)) * 1/xq.size
        L = 10*l1 + l2 + 100*l3

        return keras.Model(inputs=[xq,xp,xn], outputs=L)
    def convfilter(self, input, nb_filte):
        # Function for convolutions that change # of filters with relu activation and batchnorm
        x = layers.Conv3D(nb_filter,
                          strides=1,
                          kernal_size=3,
                          padding='same',
                          activation='relu',
                          data_format='channels-first')(input)
        x = layers.BatchNormalization(axis=1, epsilon=1.1e-5)(x)
        return x
    def convdim(self, input, nb_filter):
        # Function for convolutions that decrease input dimensions with relu activation and batchnorm
        x = layers.Conv3D(nb_filter,
                          strides=2,
                          kernal_size=2,
                          padding='same',
                          activation='relu',
                          data_format='channels-first')(input)
        x = layers.BatchNormalization(axis=1, epsilon=1.1e-5)(x)
        return x
    def upsample(self, input, nb_filter):
        # Function for decoder convolutions that increase input dimensions with relu activation and batchnorm
        x = layers.Conv3DTranspose(nb_filter,
                                   strides=2,
                                   kernel_size=2,
                                   padding='same',
                                   activation='relu',
                                   data_format='channels_first')
        x = layers.BatchNormalization(axis=1, epsilon=1.1e-5)(x)
        return x
    def encoder(self, shape):
        #encoder
        hash_bits = 32 # Length of hash code
        inp = keras.Input(shape=shape) # Preparation of input
        l1 = self.convfilter(inp, 32)
        l2 = self.convdim(l1, 32)
        l3 = self.convfilter(l2, 64)
        l4 = self.convdim(l3, 64)
        l5 = self.convfilter(l4, 128)
        shape_restore = l5.get_shape().as_list()[1:5] # Saves shape after final encoder convolution for restoration in decoder
        self.shape_restore = shape_restore
        units_restore = shape_restore[0] * shape_restore[1] * shape_restore[2] * shape_restore[3]
        self.units_restore = units_restore
        l6 = layers.Flatten(data_format = 'channels_first')(l5) # Flattening for dense layer
        self.hash = layers.Dense(hash_bits, activation='sigmoid', name='l7')(l6) # Sigmoid layer with units=hash code length
        out = self.hash
        return keras.Model(inputs=inp, outputs=out, name='encoder')
    def decoder(self):
        #decoder
        hash = self.hash
        l8 = layers.Dense(self.units_restore, activation = 'relu')(hash) # Restores previous # of units
        l9 = layers.Reshape((self.shape_restore[0], self.shape_restore[1], self.shape_restore[2], self.shape_restore[3]))(l8) # Restores previous shape
        l10 = self.convfilter(l9, 64)
        l11 = self.upsample(l10,64)
        l12 = self.convfilter(l11, 32)
        l13 = self.upsample(l12, 32)
        self.decoded = convup(l13, 1)
        out = self.decoded
        return keras.Model(inputs= hash, outputs = out, name = 'decoder')