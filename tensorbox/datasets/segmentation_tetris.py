import tensorflow as tf
import numpy as np
import random


class TetrisObject(object):
    """ Base class for tetris object """

    def __init__(self, env, h, w):
        self.h = h
        self.w = w
        self.env = env
        self.color = None

        self.y = np.random.randint(0, env.H - h + 1)
        self.x = np.random.randint(0, env.W - w + 1)
        self.bounding_box = [self.x, self.y, self.w, self.h]

    def get_mask(self):
        mask = np.full((self.env.H, self.env.W), False)
        mask[self.y: self.y + self.h, self.x: self.x + self.w] = True
        return mask


class SquareObject(TetrisObject):
    def __init__(self, env):
        h = w = np.random.randint(10, 20)
        TetrisObject.__init__(self, env, h, w)
        self.label = "Square"


class LineObject(TetrisObject):
    def __init__(self, env):
        if np.random.random() >= 0.5:
            h = 4
            w = np.random.randint(10, 20)
        else:
            h = np.random.randint(10, 20)
            w = 4
        TetrisObject.__init__(self, env, h, w)
        self.label = "Line"


class ImageGenerator(object):
    """ holds information about environment """

    def __init__(self, height, width, noise=0.3):
        self.H = height
        self.W = width
        self.noise = noise
        self.field = np.zeros((self.H, self.W, 3), dtype=np.float32)  # 3-channel image (RGB)
        self.objects = [
          'SquareObject(self)',
          'LineObject(self)'
        ]
        self.n_classes = len(self.objects) + 1  # count background as separate class
        self.colors = [('blue', 0, 0, 1.),
                       ('red', 1., 0, 0),
                       ('yellow', 1., 1., 0),
                       ('magenta', 1., 0, 1.),
                       ('green', 0, 1., 0),
                       ('aqua', 0, 1., 1.)]

    def get_random_image(self):
        """ creates and returns an image containing a random number of tetris objects """
        image = self.field.copy()
        image = self.create_random_noise(image)

        bboxes = []  # return list of labels
        gt = np.zeros(shape=(self.H, self.W, self.n_classes), dtype=np.float32)
        gt[:, :, 0] = 1.   # set class 0 to be background
        number_of_objects = random.randint(2, 6)

        for i in range(number_of_objects):
          image, object, class_label = self.place_random_object(image)
          gt[object.get_mask(), class_label] = 1.  # set pixels in class
          gt[object.get_mask(), 0] = 0.  # reset pixels in background channel
          bbox = (object.label, object.color, object.bounding_box)
          bboxes.append(bbox)
        return image, gt, bboxes

    def create_random_noise(self, image):
        """ add random background noise to image """
        for y in range(image.shape[0]):
          for x in range(image.shape[1]):

            if self.noise >= random.random():
              color,r,g,b = random.choice(self.colors)
              image[y,x] = (r,g,b)
        return image

    def place_random_object(self, image):
        """ add random tetris object to image """

        class_label = np.random.randint(1, self.n_classes)  # n_classes includes background
        obj = eval(self.objects[class_label-1])
        obj.color, r, g, b = random.choice(self.colors)
        image[obj.get_mask(), :] = (r, g, b)

        return image, obj, class_label


def create_tetris_dataset(train_val_split=0.8,
                          size=256,
                          height=128,
                          width=128,
                          batch_size=4,
                          noise=0.1,
                          buffer_size=16,
                          apply_preprocessing=True):

    gen = ImageGenerator(noise=noise, height=height, width=width)
    n_classes = gen.n_classes

    images = np.zeros((size, height, width, 3), dtype=np.float32)
    ground_truths = np.zeros((size, height, width, n_classes), dtype=np.float32)

    for idx in range(size):
        im, gt, bounding_boxes = gen.get_random_image()
        images[idx] = im
        ground_truths[idx] = gt

    train_size = int(train_val_split*size)

    ds_train = tf.data.Dataset.from_tensor_slices((images[:train_size], ground_truths[:train_size]))
    ds_val = tf.data.Dataset.from_tensor_slices((images[train_size:], ground_truths[train_size:]))

    if apply_preprocessing:
        ds_train = ds_train.batch(batch_size).prefetch(buffer_size)
        ds_val = ds_val.batch(batch_size)

    return ds_train, ds_val

