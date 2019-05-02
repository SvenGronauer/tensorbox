import tensorflow as tf
from tensorflow.python import layers, keras


class LeNet(keras.Model):
    def __init__(self, filters=32, kernel_size=3, pool_size=2):
        super(LeNet, self).__init__()
        self.conv_1_1 = layers.Conv2D(filters, kernel_size, activation='relu')
        self.conv_1_2 = layers.Conv2D(filters, kernel_size, activation='relu')

        self.max_pool = layers.MaxPooling2D(pool_size, pool_size)
        self.conv_2_1 = layers.Conv2D(2*filters, kernel_size, activation='relu')
        self.conv_2_2 = layers.Conv2D(2*filters, kernel_size, activation='relu')

        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(256, activation='relu')
        # self.dropout = layers.Dropout(0.5)
        self.d2 = layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv_1_1(x)
        x = self.conv_1_2(x)
        x = self.max_pool(x)
        # second convolution block
        x = self.conv_2_1(x)
        x = self.conv_2_2(x)

        # dense layers
        x = self.flatten(x)
        x = self.d1(x)
        # x = self.dropout(x)
        return self.d2(x)


def get_sequential_lenet(filters=32, kernel_size=2, stride=1, pool_size=2):
    model = keras.Sequential()

    model.add(layers.Conv2D(filters, kernel_size, stride, padding='same', activation='relu'))
    model.add(layers.Conv2D(filters, kernel_size, stride, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=pool_size))

    # mid-level
    model.add(layers.Conv2D(filters*2, kernel_size, stride, padding='same', activation='relu'))
    model.add(layers.Conv2D(filters*2, kernel_size, stride, padding='same', activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))

    return model