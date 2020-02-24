from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Average, Multiply, Add
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import tensorflow as tf

import matplotlib.pyplot as plt
import scipy.misc

import sys
from glob import glob
import os
import scipy
import random
from keras import backend as K
from keras.layers import Layer, AveragePooling2D, Reshape, Flatten
from keras.models import Model

import numpy as np

def age_loss(y_true, y_pred):
	return tf.reduce_mean(tf.to_float(tf.less(y_true, 20))*tf.maximum(tf.abs(y_pred-y_true)-5, 0) + tf.to_float(tf.logical_and(tf.greater_equal(y_true,20), tf.less(y_true,55)))*tf.maximum(tf.abs(y_pred-y_true)-10, 0) + tf.to_float(tf.logical_and(tf.greater_equal(y_true,55), tf.less(y_true,85)))*tf.maximum(tf.abs(y_pred-y_true)-20, 0) + tf.to_float(tf.greater_equal(y_true,85))*tf.maximum(tf.abs(y_pred-y_true)-30, 0))

def race_loss(y_true, y_pred):
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

class Conv2DFilterGen(Layer):

    def __init__(self, filter_size, channels_in, channels_out, **kwargs):
        self.filter_size = filter_size
        self.channels_in = channels_in
        self.channels_out = channels_out
        super(Conv2DFilterGen, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel_1 = self.add_weight(name='kernel_1', shape=(input_shape[1], input_shape[1]), initializer='uniform', trainable=True)
        self.kernel_2 = self.add_weight(name='kernel_2', shape=(input_shape[1], 1), initializer='uniform', trainable=True)
        self.kernel_3 = self.add_weight(name='kernel_3', shape=(1, self.filter_size*self.filter_size*self.channels_in*self.channels_out), initializer='uniform', trainable=True)
        super(Conv2DFilterGen, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        x = K.reshape(K.mean(x, axis=0), (1, x.shape[1]))
        return K.reshape(K.tanh(K.dot(K.dot(K.relu(K.dot(x, self.kernel_1)), self.kernel_2), self.kernel_3)), (self.filter_size, self.filter_size, self.channels_in, self.channels_out))

    def compute_output_shape(self, input_shape):
        return (self.filter_size, self.filter_size, self.channels_in, self.channels_out)

class FixedWeightConv2D(Layer):

    def __init__(self, **kwargs):
        super(FixedWeightConv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(FixedWeightConv2D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        a, b = x
        print(a, b)
        return K.conv2d(a, b, padding='same')

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return (shape_a[0], shape_a[1], shape_a[2], shape_b[-1])

class ParaGAN():
	
	def __init__(self, g_num_layers=3, g_init_frames=512, g_init_dim=4, d_num_layers=3, d_final_frames=512):
		# Input shape
		self.g_num_layers = g_num_layers
		self.g_init_frames = g_init_frames
		self.g_init_dim = g_init_dim
		self.d_num_layers = d_num_layers
		self.d_final_frames = d_final_frames		
		self.img_rows = g_init_dim * 2**g_num_layers
		self.img_cols = g_init_dim * 2**g_num_layers
		self.channels = 3
		self.img_shape = (self.img_rows, self.img_cols, self.channels)
		self.latent_dim = 256
		self.num_features = 6
		

		optimizer = Adam(0.0002, 0.5, clipnorm=0.2)
		
		# Build the generator
		self.generator = self.build_generator(g_num_layers, g_init_frames, g_init_dim)
		
		# Build the discriminator
		self.discriminator = self.build_discriminator(d_num_layers, d_final_frames)
		self.discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)
		
		# Build the age estimator
		self.age_estimator = self.build_age_estimator()
		self.age_estimator.load_weights('checkpoints/agenet-05-1.63.hdf5')
		self.age_estimator.trainable = False
		
		# Build the race estimator
		self.race_estimator = self.build_race_estimator()
		self.race_estimator.load_weights('checkpoints/racenet-05-0.76.hdf5')
		self.race_estimator.trainable = False
		
		# The generator takes noise as input and generates imgs
		features = Input(shape=(self.num_features,))
		latent_vec = Input(shape=(self.latent_dim,))
		img = self.generator([features, latent_vec])
		age = self.age_estimator(img)
		race = self.race_estimator(img)
		self.discriminator.trainable = False
		valid = self.discriminator(img)
		
		self.combined = Model([features, latent_vec], [valid, age, race])
		self.combined.compile(loss=["binary_crossentropy", age_loss, race_loss], optimizer=optimizer, loss_weights=[0.5, 0.5, 0.])


	def build_generator(self, g_num_layers, g_init_frames, g_init_dim):

		features = Input(shape=(self.num_features,))
		latent_vec = Input(shape=(latent_dim,))

		x = Dense(g_init_frames * g_init_dim * g_init_dim, activation="relu", input_dim=self.latent_dim)(latent_vec)
		x = Reshape((g_init_dim, g_init_dim, g_init_frames))(x)
		
		frames = g_init_frames
		
		for _ in range(g_num_layers):
			x = UpSampling2D()(x)
			conv = Conv2DFilterGen(5, frames, int(frames/2))(features)
			x = FixedWeightConv2D()([x, conv])
			x = Activation('relu')(x)
			x = BatchNormalization(momentum=0.8)(x)
			frames = int(frames/2)
		
		conv = Conv2DFilterGen(5, frames, self.channels)(features)
		img = FixedWeightConv2D()([x, conv])

		return Model([features, latent_vec], img)
	
	def build_discriminator(self, d_num_layers, d_final_frames):

		model = Sequential()
		
		frames = int(d_final_frames/2**d_num_layers)
		
		model.add(Conv2D(frames, kernel_size=3, strides=2, padding="same", input_shape=(self.img_rows, self.img_cols, self.channels)))
		model.add(LeakyReLU(0.2)
		
		for _ in range(d_num_layers-1):
			model.add(Conv2D(frames, kernel_size=3, strides=2, padding="same"))
			model.add(LeakyReLU(0.2))
			model.add(BatchNormalization(momentum=0.8))
			frames = frames * 2
		
		model.add(Flatten())
		model.add(Dense(1, activation="sigmoid"))


		img = Input(shape=(self.img_rows, self.img_cols, self.channels))
		valid = model(img)

		return Model(img, valid)
		
	def build_age_estimator(self):
		
		x_age = Input(shape=self.img_shape)
		conv1_age = Conv2D(32, (3,3), padding="same")(x_age)
		maxp1_age = MaxPooling2D(strides=(2,2))(conv1_age)
		conv2_age = Conv2D(64, (3,3), padding="same")(maxp1_age)
		maxp2_age = MaxPooling2D(strides=(2,2))(conv2_age)
		conv3_age = Conv2D(128, (3,3), padding="same")(maxp2_age)
		maxp3_age = MaxPooling2D(strides=(2,2))(conv3_age)
		l1_age = Reshape((32768,))(maxp1_age)
		l2_age = Reshape((16384,))(maxp2_age)
		l3_age = Reshape((8192,))(maxp3_age)
		fc_age = Concatenate()([l1_age, l2_age, l3_age])
		age = Dense(1)(fc_age)
		
		return Model(inputs=x_age, outputs=age)
		
	def build_race_estimator(self):
		
		x_race = Input(shape=self.img_shape)
		conv1_race = Conv2D(32, (3,3), padding="same")(x_race)
		maxp1_race = MaxPooling2D(strides=(2,2))(conv1_race)
		conv2_race = Conv2D(64, (3,3), padding="same")(maxp1_race)
		maxp2_race = MaxPooling2D(strides=(2,2))(conv2_race)
		conv3_race = Conv2D(128, (3,3), padding="same")(maxp2_race)
		maxp3_race = MaxPooling2D(strides=(2,2))(conv3_race)
		l1_race = Reshape((32768,))(maxp1_race)
		l2_race = Reshape((16384,))(maxp2_race)
		l3_race = Reshape((8192,))(maxp3_race)
		fc_race = Concatenate()([l1_race, l2_race, l3_race])
		race_logits = Dense(5)(fc_race)
		
		return Model(inputs=x_race, outputs=race_logits)
