import os
import sys
import gzip
import logging

import numpy as np

logger = logging.getLogger(__name__)

class MNISTWrapper(object):
  def __init__(self, path_to_mnist_files, reshape_to_images=True):
    '''
    Creates a MNIST Wrapper instance.

    path_to_mnist_files   string
                          The path to the folder where the four .gz files are located

    reshape_to_images     bool (default True)
                          If true, the data is reshaped to a batch of images. The shape                           
                          is in this case (batch, channel = 1, height = 28, width = 28)
                          Otherwise the images remain flattened, thus the shape of the data
                          is (batch, 28 x 28)                                   
    '''

    raise NotImplementedError("Check the mnist wrapper first")

    filenames = {}

    filenames["test_image"] = os.path.join(path_to_mnist_files, 't10k-images-idx3-ubyte.gz')
    filenames["test_label"] = os.path.join(path_to_mnist_files, 't10k-labels-idx1-ubyte.gz')

    filenames["train_image"] = os.path.join(path_to_mnist_files, 'train-images-idx3-ubyte.gz')
    filenames["train_label"] = os.path.join(path_to_mnist_files, 'train-labels-idx1-ubyte.gz')

    assert all(map(os.path.exists, filenames.itervalues())), "One of the files is missing"

    X_train_all = self.load_mnist_images(filenames["train_image"], reshape_to_images)
    Y_train_all = self.load_mnist_labels(filenames["train_label"])

    self.X_test = self.load_mnist_images(filenames["test_image"], reshape_to_images)
    self.Y_test_raw = self.load_mnist_labels(filenames["test_label"]).reshape(-1,1)
    self.Y_test = self.one_hot_targets(self.Y_test_raw)    

    where_to_split = 10000 #10k

    self.X_train = X_train_all[:-where_to_split,...]
    self.Y_train_raw = Y_train_all[:-where_to_split,...].reshape(-1,1)
    self.Y_train = self.one_hot_targets(self.Y_train_raw)

    self.X_val = X_train_all[-where_to_split:,...]
    self.Y_val_raw = Y_train_all[-where_to_split:,...].reshape(-1,1)
    self.Y_val = self.one_hot_targets(self.Y_val_raw)

    self.N_test = self.Y_test.shape[0]
    self.N_train = self.Y_train.shape[0]
    self.N_val = self.Y_val.shape[0]

    return
  def __len__(self):
    ''' Returns the length of the trainings data set '''
    return self.N_train

  def load_mnist_images(self,filename, reshape_to_images):
    ''' Loads the images from the given file '''
        
    # Read the inputs in Yann LeCun's binary format. The data is then
    # a long vector
    with gzip.open(filename, 'rb') as f:
      data = np.frombuffer(f.read(), np.uint8, offset=16)
    
    # Reshape to monochrome 2D images, following the shape convention: 
    # (batch, channels, rows, columns)
    if reshape_to_images:
      data = data.reshape(-1, 1, 28, 28)

    # Reshape to batch of image vectors, every row is one flattened image
    else:
      data = data.reshape(-1, 28 * 28)
    #end

    # The inputs come as bytes, we convert them to float32 in range [0,1].
    # (Actually to range [0, 255/256], for compatibility to the version
    # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
    return data / np.float32(256)
  def load_mnist_labels(self,filename):
    ''' Loads the labels from the given file '''
    
    # Read the labels in Yann LeCun's binary format.
    with gzip.open(filename, 'rb') as f:
      data = np.frombuffer(f.read(), np.uint8, offset=8)
    
    # The labels are vectors of integers now, that's exactly what we want.
    return data         
  def one_hot_targets(self, Y):
    '''
    Converts the arrays of class labels 0,1 to a one hot encoded representation
    '''
    assert Y.ndim == 2 and Y.shape[1] == 1, "Only 1-dimensional target values can be one-hot encoded"

    # Number of class labels is the largest number plus one in the training label set
    # +1 required because class labels start at zero
    classes = Y.max() + 1
    samples = Y.shape[0]

    Y_onehot = np.zeros((samples, classes))
    Y_onehot[np.arange(samples),Y[:,0]] = 1
    
    return Y_onehot

  def print_test(self, index):
    ''' Prints the specified testing image '''

    if not 0 <= index < self.N_test:
      raise IndexError("The index {} is out of range.".format(index))

    logger.info("Printing test image {} - it is a {}".format(index, self.Y_test[index]))
    
    print self.display(self.X_test[index,...])
  
    return
  def print_train(self, index):
    ''' Prints the specified training image '''
    if not 0 <= index < self.N_train:
      raise IndexError("The index {} is out of range.".format(index))

    logger.info("Printing training image {} - it is a {}".format(index, self.Y_train[index]))    
    print self.display(self.X_train[index,...])

    return
  def print_val(self, index):
    ''' Prints the specified validation image '''

    if not 0 <= index < self.N_val:
      raise IndexError("The index {} is out of range.".format(index))

    logger.info("Printing validation image {} - it is a {}".format(index, self.Y_val[index]))    
    print self.display(self.X_val[index,...])

    return
  def display(self, img, threshold=0.25):
    '''
    Prepares the image for printing to the console.
    If the pixel value in an image lies above threshold a '@' is printed
    otherwise a dot '.'
    '''
    render = ''
    
    # Flattened image has to be reshaped for printing
    if len(img.shape) == 1:
      img2 = img.reshape((28,28))

    # Monochrome image -> extract from first channel
    elif len(img.shape) == 3 and img.shape[0] == 1:
      img2 = img[0,...]
    
    # Other formates are currently not supported
    else:
      raise ValueError("Unsupported image format detected: {}".format(img.shape))
    
    sx,sy = img2.shape

    for i in range(sx):
      for j in range(sy):
        render += '@' if img2[i,j] > threshold else '.'
      #end for j

      render += '\n'
    #end for i

    return render

  def get_data_shape(self):
    '''
    Returns the shape of one data sample:
      reshape_to_images = True: (1,28,28)
      reshape_to_images = False: (28*28,)
    '''
    return self.X_test.shape[1:]

  def iterate_training(self, minibatch_size):
    ''' 
    Iterates over the training set in mini batches of the specified size. 
    This function is a python iterator, meaning it should be used in a loop to iterate for a complete epoch.
    Every epoch is shuffled.
    '''
    indices = np.arange(self.N_train)

    # Works inline
    np.random.shuffle(indices)

    for i in xrange(0, self.N_train, minibatch_size):

      idx = indices[i:i + minibatch_size]

      if len(idx) != minibatch_size:
        raise StopIteration()

      data = self.X_train[idx]
      label = self.Y_train[idx]

      yield data, label
    #end for
    
    raise StopIteration
  def iterate_testing(self, minibatch_size):
    ''' 
    Iterates over the testing set in mini batches of the specified size. 
    This function is a python iterator, meaning it should be used in a loop to iterate for a complete epoch.    
    '''

    indices = np.arange(self.N_test)

    for i in xrange(0, self.N_test, minibatch_size):

      idx = indices[i:i + minibatch_size]

      if len(idx) != minibatch_size:
        raise StopIteration()

      data = self.X_test[idx]
      label = self.Y_test[idx]

      yield data, label
    #end for
    
    raise StopIteration
  def iterate_validate(self, minibatch_size):
    ''' 
    Iterates over the testing set in mini batches of the specified size. 
    This function is a python iterator, meaning it should be used in a loop to iterate for a complete epoch.    
    '''

    indices = np.arange(self.N_val)

    for i in xrange(0, self.N_val, minibatch_size):

      idx = indices[i:i + minibatch_size]

      if len(idx) != minibatch_size:
        raise StopIteration()

      data = self.X_val[idx]
      label = self.Y_val[idx]

      yield data, label
    #end for
    
    raise StopIteration

if __name__ == "__main__":

  wrapper = MNISTWrapper("./mnist_data_set")

  print "All done"
