from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Average, Multiply, Add
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import scipy.misc

import sys
from glob import glob
import os
import scipy
import random
#import cv2

import numpy as np

class GAN():
	
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
		

		optimizer = Adam(0.0002, 0.5, clipnorm=0.2)
		
		# Build the generator
		self.generator = self.build_generator(g_num_layers, g_init_frames, g_init_dim)
		
		# Build the discriminator
		self.discriminator = self.build_discriminator(d_num_layers, d_final_frames)
		self.discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)
		
		# The generator takes noise as input and generates imgs
		z = Input(shape=(self.latent_dim,))
		img = self.generator(z)
		self.discriminator.trainable = False
		valid = self.discriminator(img)
		
		self.GAN = Model(z, valid)
		self.GAN.compile(loss="binary_crossentropy", optimizer=optimizer)


	def build_generator(self, g_num_layers, g_init_frames, g_init_dim):

		model = Sequential()

		model.add(Dense(g_init_frames * g_init_dim * g_init_dim, activation="relu", input_dim=self.latent_dim))
		model.add(Reshape((g_init_dim, g_init_dim, g_init_frames)))
		
		frames = g_init_frames
		
		for _ in range(g_num_layers):
			model.add(UpSampling2D())
			model.add(Conv2D(frames, kernel_size=3, padding="same", activation="relu"))
			model.add(BatchNormalization(momentum=0.8))
			model.add(Conv2D(frames, kernel_size=3, padding="same", activation="relu"))
			model.add(BatchNormalization(momentum=0.8))
			frames = int(frames/2)
		
		model.add(Conv2D(self.channels, kernel_size=3, padding="same", activation="tanh"))

		noise = Input(shape=(self.latent_dim,))
		img = model(noise)

		return Model(noise, img)
		
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
	

	def train(self, epochs=1, n_discriminator_update=1, n_generator_update=2, batch_size=64, save_interval=500):

		data = glob(os.path.join("../faces/*"))
		
		for epoch in range(epochs):
			random.shuffle(data)
			for idx in range(int(len(data)/batch_size)):
				batch_files = data[idx*batch_size:(idx+1)*batch_size]
				batch = [(np.array(scipy.misc.imresize(scipy.misc.imread(batch_file), (self.img_rows, self.img_cols))) for batch_file in batch_files]
				batch = [x/127.5 - 1 for x in batch]
				batch = np.array(batch).astype(np.float32)
				batch_z = np.zeros((batch_size, self.latent_dim)).astype(np.float32)
				for i in range(batch_size):
					np.random.seed()
					batch_z[i,:] = np.random.uniform(-1,1, self.latent_dim).astype(np.float32)
				
				#print(batch.shape, batch_z.shape)
				# Update D network
				for _ in range(n_discriminator_update):
					imgs = self.generator.predict(batch_z)
					errD_fake = self.discriminator.train_on_batch(batch, np.ones(batch_size))
					errD_real = self.discriminator.train_on_batch(imgs, np.zeros(batch_size))

				# Update G network
				for _ in range(n_generator_update):
					errG = self.GAN.train_on_batch(batch_z, np.ones(batch_size))
				
				print("Epoch {:2d} [{:2d}]/[{:2d}]: errD_fake = {:.4f}, errD_real = {:.4f}, errG = {:.4f}".format(epoch, idx, int(len(data)/batch_size), errD_fake, errD_real, errG))


if __name__ == '__main__':
	for d_num_layers in [2, 3, 4]:
		for g_num_layers in [2, 3, 4]:
			for d_final_frames in [128, 256, 512]:
				for g_init_frames in [128, 256, 512]:
					gan = GAN(d_num_layers=d_num_layers, g_num_layers=g_num_layers, d_final_frames=d_final_frames, g_init_frames=g_init_frames)
					gan.train()