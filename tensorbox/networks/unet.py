import tensorflow as tf
from tensorflow.python import layers, keras  # Fix for TF2.0a and PyCharm
from tensorbox.networks import BaseNetwork


class DoubleConvolution(layers.Layer):
    """ applies two 2D-convolutions each followed by ReLU activations"""

    def __init__(self, output_filters=32, kernel_size=3):
        super(DoubleConvolution, self).__init__()
        self.c1 = layers.Conv2D(output_filters, kernel_size, activation='relu', padding='same')
        self.c2 = layers.Conv2D(output_filters, kernel_size, activation='relu', padding='same')

    def call(self, x, **kwargs):
        x = self.c1(x)
        return self.c2(x)


class DownConvolution(layers.Layer):
    """  Layer that performs 2 convolutions and 1 MaxPool """

    def __init__(self, output_filters, pooling=True, stride=2):
        super(DownConvolution, self).__init__()
        self.double_convolution = DoubleConvolution(output_filters=output_filters, kernel_size=3)
        self.pooling = pooling
        if pooling:
            self.pool = layers.MaxPooling2D(pool_size=stride, strides=stride)

    def call(self, x, **kwargs):
        x = self.double_convolution(x)
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConvolution(layers.Layer):
    """  Layer that performs 2 convolutions and 1 UpConvolution """

    def __init__(self, output_filters, stride=2):
        super(UpConvolution, self).__init__()
        self.double_convolution = DoubleConvolution(output_filters=output_filters, kernel_size=3)
        self.up_convolution = layers.Conv2DTranspose(filters=output_filters,
                                                     kernel_size=3,
                                                     strides=(stride, stride),
                                                     padding='same')

    def call(self, inputs, **kwargs):
        from_down, from_up = inputs
        x = self.up_convolution(from_up)
        merged = tf.concat((from_down, x), axis=3)
        # x = self.up_convolution(x)
        x = self.double_convolution(merged)
        return x


class UNet(keras.Model, BaseNetwork):
    """ Implementation of U-Net
        authors: Ronneberger - U-Net CNNs for Biomedical Image Segmentation - 2015
        https://arxiv.org/pdf/1505.04597.pdf
    """
    def __init__(self, num_filters, num_classes, depth=3):
        super(UNet, self).__init__()
        self.num_filters = num_filters
        self.depth = depth
        self.num_classes = num_classes

        self.final_convolution = layers.Conv2D(num_classes, kernel_size=3, activation='softmax', padding='same')
        self.bottom_convolution = DownConvolution(output_filters=num_filters * 2**depth, pooling=False)

        # encoder path
        self.down_convolutions = []
        for i in range(depth):
            outs = num_filters*(2**i)
            self.down_convolutions.append(DownConvolution(output_filters=outs, pooling=True))

        # decoder path
        self.up_convolutions = []
        for i in range(depth):
            outs = num_filters * 2**i
            self.up_convolutions.append(UpConvolution(output_filters=outs))

    def call(self, x, **kwargs):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, dc in enumerate(self.down_convolutions):
            x, before_pool = dc(x)
            encoder_outs.append(before_pool)

        x, _ = self.bottom_convolution(x)
        # TODO tf concat down and up paths

        for i in reversed(range(self.depth)):

            before_pool = encoder_outs[i]
            uc = self.up_convolutions[i]
            x = uc([before_pool, x])
            # merged = tf.concat((before_pool, x), axis=3, name='concat_{}'.format(i))
            # uc = self.up_convolutions[i]
            # x = uc(merged)

        x = self.final_convolution(x)
        return x
