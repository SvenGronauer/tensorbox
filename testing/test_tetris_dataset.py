import tensorflow as tf
from toolbox.datasets.segmentation_tetris import create_tetris_dataset
import unittest


class TestUNet(unittest.TestCase):

    def test_random_tetris_set(self):
        train_val_split = 0.8
        size = 256
        height = 128
        width = 128
        batch_size = 4
        noise = 0.1
        buffer_size = 128
        apply_preprocessing = True
        train_set, val_set = create_tetris_dataset(train_val_split=train_val_split,
                                                   size=size,
                                                   height=height,
                                                   width=width,
                                                   batch_size=batch_size,
                                                   noise=noise,
                                                   buffer_size=buffer_size,
                                                   apply_preprocessing=apply_preprocessing)

        self.assertTrue(isinstance(train_set, tf.data.Dataset))
        self.assertTrue(isinstance(val_set, tf.data.Dataset))


if __name__ == '__main__':
    print(tf.__version__)
    unittest.main()
