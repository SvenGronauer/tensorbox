import numpy as np
import pickle
import os

import logging
logger = logging.getLogger(__name__)


class BaseWrapper(object):
    def __init__(self, T, data_name, use_file=False, force_use_file=False, recreate=False, *args, **kwargs):
        '''
        Initialization of the base class for all wrappers
        '''

        self.build_dir = kwargs.get("build_dir", '/var/tmp/data')
        self.dtype = kwargs.get("dtype", np.float64)

        self.data_name = os.path.join(self.build_dir,data_name)

        self.use_file = use_file
        self.force_use_file = force_use_file
        self.recreate = recreate

        if self.force_use_file and not self.use_file:
            raise ValueError("You cannot enforce the usage of files with allowing the usage of files at all!")

        if not os.path.exists(self.build_dir):
            os.mkdir(self.build_dir)
        #end if

        self.X_train, self.Y_train = None, None
        self.X_val, self.Y_val = None, None
        self.X_test, self.Y_test = None, None

        self.N_test = None
        self.N_train = None
        self.N_val = None

        self.T = int(T)

        self._dataset_created = False

        if self.use_file:
            if not self.recreate:
                success = self._load_from_file()
            else:
                success = False
                logger.warning("The possibly existing file has been ignored, reason: recreate = True")
            #end
        else:
            success = False
        #end

        if success:
            logger.info("Restored data set from file.")
        else:
            if force_use_file:
                raise RuntimeError("The data wrapper would create a new data set, but an existing must be used! Reason: force_use_file = True")

            self._create_data_set(self.T)
            logger.info("Created a new data set.")

            if self.use_file:
                success = self._save_to_file()
                if success:
                    logger.info("Saved data set to file.")
                else:
                    logger.warning("Saving the dataset to file failed, reason: {}".format(e))
                #end
            #end
        #end

        self._dataset_created = True

        self._amount_data_from_shape()

        if self.T != self.N_train:
            msg = "The desired size of the data set T={} does not match the actual one: {}. Restoring a file by accident?"
            raise RuntimeError(msg.format(self.T, self.N_train))

        return

    def __len__(self):
        ''' Returns the length of the trainings data set '''
        return self.N_train

    def _amount_data_from_shape(self):
        '''
        Extracts the number of test, train and validation samples from the shape information.
        Asserts that the data set has been created successfully
        '''
        assert self._dataset_created, "Call this function only after the creation of a data set"

        self.N_test = self.Y_test.shape[0]
        self.N_train = self.Y_train.shape[0]
        self.N_val = self.Y_val.shape[0]

        return
    
    def _create_data_set(self, T):
        ''' Implement this function to create or load the actual data set '''
        raise NotImplementedError()

    def _serialize(self):
        ''' Shall return what needs to be stored in the pickle file '''
        raise NotImplementedError()
    
    def _deserialize(self, result):
        ''' Gets the result of loading a pickle file and shall parse that information and update self '''
        raise NotImplementedError()

    def _load_from_file(self):
        ''' Returns the data set stored in the build directory '''

        try:
            with open(self.data_name, "rb") as the_file:
                result = pickle.load(the_file)
            #end with

            self._deserialize(result)

            success = True

        except IOError as e:
            success = False

        return success
    
    def _save_to_file(self):
        ''' Stores the current data set in the build directory '''

        # Create the matlab version for debugging
        # scipy.io.savemat('data_4r_1k_random.mat', dict(X_in=self.X_train, Y_out=self.Y_train))

        try:
            with open(self.data_name, "wb") as the_file:
                pickle.dump(self._serialize(), the_file, pickle.HIGHEST_PROTOCOL)
            #end with

            success = True

        except IOError as e:
            success = False

        return success

    def get_data_shape(self):
        ''' Returns the shape of one data sample '''
        return self.X_test.shape[1:]
    
    def sort_data_set(self, X,Y):
        ''' Call this function with a data set (X,Y) to get a sorted version back (label is index of list) '''
        assert Y.ndim == 2 and Y.shape[1] == 1, "Only 1-dimensional target values can be sorted"

        # Number of class labels is the largest number plus one in the training label set
        # +1 required because class labels start at zero
        classes = np.max(Y) + 1

        X_sorted = [[] for _ in range(classes)]

        for x,y in zip(X,Y):
            X_sorted[y[0]].append(x)

        return map(np.array, X_sorted)
    
    def one_hot_targets(self, Y):
        '''
        Converts the arrays of class labels 0,1 to a one hot encoded representation
        '''
        assert Y.ndim == 2 and Y.shape[1] == 1, "Only 1-dimensional target values can be one-hot encoded"

        # Number of class labels is the largest number plus one in the training label set
        # +1 required because class labels start at zero
        classes = Y.max() + 1
        samples = Y.shape[0]

        Y_onehot = np.zeros((samples, classes), self.dtype)
        Y_onehot[np.arange(samples), Y[:,0]] = 1

        return Y_onehot

    def sample_training(self, minibatch_size):
        '''
        Sample a minibatch from the training set
        '''

        indices = np.random.choice(self.N_train, minibatch_size, replace=False)

        data = self.X_train[indices]
        label = self.Y_train[indices]

        return data, label
    
    def sample_testing(self, minibatch_size):
        '''
        Sample a minibatch from the testing set
        '''

        indices = np.random.choice(self.N_test, minibatch_size, replace=False)

        data = self.X_test[indices]
        label = self.Y_test[indices]

        return data, label
    
    def sample_validate(self, minibatch_size):
        '''
        Sample a minibatch from the validation set
        '''

        indices = np.random.choice(self.N_val, minibatch_size, replace=False)

        data = self.X_val[indices]
        label = self.Y_val[indices]

        return data, label

    def iterate_training(self, minibatch_size):
        '''
        Iterates over the training set in mini batches of the specified size.
        This function is a python iterator, meaning it should be used in a loop to iterate for a complete epoch.
        Every epoch is shuffled.
        '''
        indices = np.arange(self.N_train)

        # Works inline
        np.random.shuffle(indices)

        for i in range(0, self.N_train, minibatch_size):

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

        for i in range(0, self.N_val, minibatch_size):

            idx = indices[i:i + minibatch_size]

            if len(idx) != minibatch_size:
                raise StopIteration()

            data = self.X_val[idx]
            label = self.Y_val[idx]

            yield data, label
        #end for

        raise StopIteration
