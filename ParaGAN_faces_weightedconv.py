from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Average, Multiply, Add
from keras.layers import RepeatVector
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

import numpy as np

import keras.backend as K

def age_loss(y_true, y_pred):
	return tf.reduce_mean(tf.to_float(tf.less(y_true, 20))*tf.maximum(tf.abs(y_pred-y_true)-5, 0) + tf.to_float(tf.logical_and(tf.greater_equal(y_true,20), tf.less(y_true,55)))*tf.maximum(tf.abs(y_pred-y_true)-10, 0) + tf.to_float(tf.logical_and(tf.greater_equal(y_true,55), tf.less(y_true,85)))*tf.maximum(tf.abs(y_pred-y_true)-20, 0) + tf.to_float(tf.greater_equal(y_true,85))*tf.maximum(tf.abs(y_pred-y_true)-30, 0))

def race_loss(y_true, y_pred):
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

class ParaGAN():
	
	def __init__(self):
		# Input shape
		self.g_num_layers = g_num_layers
		self.g_init_frames = g_init_frames
		self.g_init_dim = g_init_dim		
		self.img_rows = g_init_dim * 2**g_num_layers
		self.img_cols = g_init_dim * 2**g_num_layers
		self.channels = 3
		self.img_shape = (self.img_rows, self.img_cols, self.channels)
		self.latent_dim = 256
		self.num_features = 6
		self.mod_tensors = []

		optimizer = Adam(0.0002, 0.5, clipnorm=0.2)
		
		
		# Build the generator models
		self.generator_models = self.build_generator_models(g_num_layers, g_init_frames, g_init_dim)
		i = 0
		for model in self.generator_models:
			model.load_weights('checkpoints/face_generator_'+str(i)+'.hdf5')
			model.trainable = False
			i += 1
		#Build the age estimator
		self.age_estimator = self.build_age_estimator()
		self.age_estimator.load_weights('checkpoints/agenet-05-1.63.hdf5')
		self.age_estimator.trainable = False
		#Build the race estimator
		self.race_estimator = self.build_race_estimator()
		self.race_estimator.load_weights('checkpoints/racenet-05-0.76.hdf5')
		self.race_estimator.trainable = False
		
		features = Input(shape=(self.num_features,))
		latent_vec = Input(shape=(self.latent_dim,))
		#Generate models for tensor value modifications
		self.param_modifiers = [self.build_parametric_modifier(self.latent_dim, None)] + [self.build_parametric_modifier(m.layers[-1].output.shape[-1], m.layers[-1].output.shape[1:]) for m in models[self.mod_tensors]]
		#Build the parametric generator
		self.paragen = self.build_parametric_generator()
		img = self.paragen([features, latent_vec])
		age = self.age_estimator(img)
		race = self.race_estimator(img)
		
		self.combined = Model([features, latent_vec], [age, race])
		self.combined.compile(loss=[age_loss, race_loss], optimizer=optimizer, loss_weights=[1., 0.])


	def build_generator_models(self, g_num_layers, g_init_frames, g_init_dim):
		
		models = []
		
		model = Sequential()
		model.add(Dense(g_init_frames * g_init_dim * g_init_dim, activation="relu", input_dim=self.latent_dim))
		model.add(Reshape((g_init_dim, g_init_dim, g_init_frames)))
		models.append(model)
		
		shape = (g_init_dim, g_init_dim, g_init_frames)
		frames = g_init_frames
		
		for _ in range(g_num_layers):
			model = Sequential()
			model.add(UpSampling2D(input_shape=shape))
			model.add(Conv2D(frames, kernel_size=3, padding="same", activation="relu"))
			model.add(BatchNormalization(momentum=0.8))
			model.add(Conv2D(frames, kernel_size=3, padding="same", activation="relu"))
			model.add(BatchNormalization(momentum=0.8))
			models.append(model)
			frames = int(frames/2)
			shape = (shape[0]*2, shape[1]*2, frames)
		
		model = Sequential()
		model.add(Conv2D(self.channels, kernel_size=3, padding="same", activation="tanh", input_shape=shape))
		models.append(model)

		return models
		
	def build_parametric_modifier(self, units, shape):
		
		features = Input(shape = (self.num_features,))
		
		x = Dense(units, acivation='tanh')(features)
		x = x * units
		
		if shape is not None:
			x = RepeatVector(np.prod(list(shape)))(x)
			x = Reshape(shape)(x)
		
		return Model(features, x)
	
	def build_parametric_generator(self):
		
		inp = Input(shape=(latent_dim,))
		x = self.param_modifiers[0](inp)
		
		for i in range(len(self.generator_models)):
			x = self.generator_models[i](x)
			if i in self.mod_tensors:
				x = self.param_modifiers[i+1](x)
		
		return Model(inp, x)
		
	
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


	def train(self, epochs, batch_size=128, save_interval=500):

		for epoch in range(epochs):
			
			# Select a random half of images
			ages = np.random.uniform(0, 125, (batch_size,1))
			race_probs_ = np.random.randint(0, 1e5, (batch_size, 5))
			race_probs = np.array([race_probs_[i] / np.max(race_probs_, 1)[i] for i in range(batch_size)])
			f_in = np.concatenate((ages, race_probs), 1)
			rand_noise = np.random.uniform(0,1,(batch_size, self.latent_dim))
			
			paragen_loss = self.combined.train_on_batch([f_in, rand_noise], [ages, race_probs])

			print (str(epoch)+" ParaGen_loss: "+str(deco_loss))
			# If at save interval => save model
			if epoch % save_interval == 0:
				#self.save_imgs(epoch)
				self.image_mod.save('checkpoints/weightedconv_image_modifier.hdf5')



if __name__ == '__main__':
	paragan = ParaGAN()
	paragan.train(epochs=4000, batch_size=32, save_interval=50)