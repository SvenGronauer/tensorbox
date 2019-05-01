import tensorflow as tf
from toolbox.networks.unet import UNet
import unittest


class TestUNet(unittest.TestCase):

    def test_random_input(self):
        num_classes = 3
        N = 5
        net = UNet(num_filters=8, num_classes=num_classes)
        dim = 28
        noise = tf.random.normal(shape=(N, dim, dim, 1))
        pred = net(noise)
        self.assertTrue(pred.shape == (N, dim, dim, num_classes), 'Output shape does not match input shape.')


if __name__ == '__main__':
    print(tf.__version__)
    unittest.main()
