from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Average, Multiply, Add
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
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

In_1 = Input(shape=(128,))
X = Dense(4*4*512)(In_1)
X = Reshape((4,4,512))(X)
X = UpSampling2D()(X)
X1 = Conv2D(256, 3, padding="same", activation="relu")(X)
Img_1 = Conv2D(3, 3, padding="same", activation="tanh")(X1)
Block_1 = Model(In_1, X1)
GAN_1 = Model(In_1, Img_1)

In_2 = Input(shape=(256,))
X = Dense(8*8*256)(In_2)
X = Reshape((8,8,256))(X)
X = UpSampling2D()(X)
X2 = UpSampling2D()(X1)
X2 = Concatenate()([X, X2])
X2 = Conv2D(256, 3, padding="same", activation="relu")(X2)
Img_2 = Conv2D(3, 3, padding="same", activation="tanh")(X2)
Block_2 = Model(In_2, X2)
GAN_2 = Model(In_2, Img_2)

In_3 = Input(shape=(512,))
X = Dense(16*16*256)(In_3)
X = Reshape((16,16,256))(X)
X = UpSampling2D()(X)
X3 = UpSampling2D()(X2)
X3 = Concatenate()([X, X3])
X3 = Conv2D(256, 3, padding="same", activation="relu")(X3)
Img_3 = Conv2D(3, 3, padding="same", activation="tanh")(X3)
Block_3 = Model(In_3, X3)
GAN_3 = Model(In_3, Img_3)

In_4 = Input(shape=(1024,))
X = Dense(8*8*256)(In_4)
X = Reshape((8,8,256))(X)
X = UpSampling2D()(X)
X4 = UpSampling2D()(X3)
X4 = Concatenate()([X, X4])
X4 = Conv2D(256, 3, padding="same", activation="relu")(X4)
Img_4 = Conv2D(3, 3, padding="same", activation="tanh")(X4)
Block_4 = Model(In_4, X4)
GAN_4 = Model(In_4, Img_4)

