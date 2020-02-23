from keras import backend as K
from keras.layers import Layer, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Reshape, Flatten, Input, Activation, UpSampling2D, BatchNormalization
from keras.models import Model
from glob import glob
import os
import scipy
import random
import numpy as np

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

img_shape = (64, 64, 3)
num_classes = 5
latent_dim = 100
Img = Input(shape = img_shape)
X = Conv2D(64, 3, padding='same', activation='relu')(Img)
X = Conv2D(64, 3, padding='same', activation='relu')(X)
X = MaxPooling2D()(X)
X = Conv2D(128, 3, padding='same', activation='relu')(X)
X = Conv2D(128, 3, padding='same', activation='relu')(X)
X = MaxPooling2D()(X)
X = Conv2D(256, 3, padding='same', activation='relu')(X)
X = Conv2D(256, 3, padding='same', activation='relu')(X)
X = MaxPooling2D()(X)
X = Flatten()(X)
X = Dense(1, activation='sigmoid')(X)

discriminator = Model(Img, X)
discriminator.compile(loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5, clipnorm=0.2))
discriminator.summary()

L = Input(shape = (latent_dim, ))
C = Input(shape = (num_classes, ))
X = Dense(1024 * 4 * 4, activation="relu")(L)
X = Reshape((4, 4, 1024))(X)
X = UpSampling2D()(X)
X = Conv2D(512, kernel_size=5, padding="same")(X)
X = BatchNormalization(momentum=0.8)(X)
X = Activation("relu")(X)
X = UpSampling2D()(X)
Conv3_1 = Conv2DFilterGen(5, 512, 256)(C)
X = FixedWeightConv2D()([X, Conv3_1])
X = BatchNormalization(momentum=0.8)(X)
X = Activation("relu")(X)
X = UpSampling2D()(X)
Conv4_1 = Conv2DFilterGen(5, 256, 128)(C)
X = FixedWeightConv2D()([X, Conv4_1])
X = BatchNormalization(momentum=0.8)(X)
X = Activation("relu")(X)
X = UpSampling2D()(X)
Conv5_1 = Conv2DFilterGen(5, 128, img_shape[2])(C)
X = FixedWeightConv2D()([X, Conv5_1])
X = Activation("tanh")(X)

generator = Model([L, C], X)
generator.summary()

discriminator.trainable = False
L = Input(shape = (latent_dim, ))
C = Input(shape = (num_classes, ))
Img = generator([L, C])
valid = discriminator(Img)
GAN = Model([L, C], valid)
GAN.compile(loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5, clipnorm=0.2))

epochs = 5
n_discriminator_update=1
n_generator_update=2
batch_size=64
save_interval=500

data = glob(os.path.join("../faces/*"))

for epoch in range(epochs):
	random.shuffle(data)
	for idx in range(int(len(data)/batch_size)):
		batch_files = data[idx*batch_size:(idx+1)*batch_size]
		batch = [scipy.misc.imresize(scipy.misc.imread(batch_file), (img_shape[0], img_shape[1])) for batch_file in batch_files]
		batch = np.array(batch).astype(np.float32)
		batch_z = np.zeros((batch_size, latent_dim)).astype(np.float32)
		batch_c = np.zeros((batch_size, num_classes)).astype(np.float32)
		for i in range(batch_size):
			np.random.seed()
			batch_z[i,:] = np.random.uniform(-1,1, latent_dim).astype(np.float32)
			batch_c[i,:] = np.random.uniform(-1,1, num_classes).astype(np.float32)
		
		#print(batch.shape, batch_z.shape)
		# Update D network
		for _ in range(n_discriminator_update):
			imgs = generator.predict(batch_z)
			errD_fake = discriminator.train_on_batch(batch, np.ones(batch_size))
			errD_real = discriminator.train_on_batch(imgs, np.zeros(batch_size))

		# Update G network
		for _ in range(n_generator_update):
			errG = GAN.train_on_batch(batch_z, np.ones(batch_size))
		
		print("Epoch {:2d} [{:2d}]/[{:2d}]: errD_fake = {:.4f}, errD_real = {:.4f}, errG = {:.4f}".format(epoch, idx, int(len(data)/batch_size), errD_fake, errD_real, errG))
		last_k_g_losses = last_k_g_losses[1:] + [errG]
		if(sum([last_k_g_losses[i] == last_k_g_losses[0] for i in range(k)]) == k):
			test = 0
			break