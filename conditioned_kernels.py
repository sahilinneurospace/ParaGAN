from keras import backend as K
from keras.layers import Layer, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Reshape, Flatten, Input, Activation
from keras.models import Model

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
num_classes = 100
Img = Input(shape = img_shape)
C = Input(shape = (num_classes, ))
X = Conv2D(64, 3, padding='same', activation='relu')(Img)
X = Conv2D(64, 3, padding='same', activation='relu')(X)
X = MaxPooling2D()(X)
X = Conv2D(128, 3, padding='same', activation='relu')(X)
X = Conv2D(128, 3, padding='same', activation='relu')(X)
X = MaxPooling2D()(X)
Conv3_1 = Conv2DFilterGen(3, 128, 256)(C)
X = FixedWeightConv2D()([X, Conv3_1])
X = Activation('relu')(X)
X = MaxPooling2D()(X)
X = Flatten()(X)
X = Dense(1, activation='sigmoid')(X)

model = Model([Img, C], X)
model.summary()