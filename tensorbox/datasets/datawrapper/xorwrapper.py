from .basewrapper import *

class XORWrapper(BaseWrapper):
  def __init__(self, *args, **kwargs):
    '''
    Creates a XOR Wrapper instance.
    '''

    # Call base class to create everything
    return super(XORWrapper, self).__init__(4, "XORDataSet.pickle", use_file = False, force_use_file = False, *args, **kwargs)

  def _create_samples_and_label_them(self, T):
    ''' Call this function to create T samples and their labels '''
    
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype = self.dtype)
    Y = np.array([0,1,1,0]).reshape(-1,1)

    return X,Y
  def _create_data_set(self, T):
    ''' Implementation of the base class, here the actual data is created '''

    self.X_train, self.Y_train_raw = self._create_samples_and_label_them(None)
    self.X_val, self.Y_val_raw = self._create_samples_and_label_them(None)
    self.X_test, self.Y_test_raw = self._create_samples_and_label_them(None)

    self.Y_train = self.one_hot_targets(self.Y_train_raw)
    self.Y_val = self.one_hot_targets(self.Y_val_raw)
    self.Y_test = self.one_hot_targets(self.Y_test_raw)

    return

  def _serialize(self):
    ''' Shall return what needs to be stored in the pickle file '''
    return self.X_train, self.Y_train_raw, self.X_val, self.Y_val_raw, self.X_test, self.Y_test_raw
  def _deserialize(self, result):
    ''' Gets the result of loading a pickle file and shall parse that information and update self '''

    self.X_train, self.Y_train_raw, self.X_val, self.Y_val_raw, self.X_test, self.Y_test_raw = result
    
    self.Y_train = self.one_hot_targets(self.Y_train_raw)
    self.Y_val = self.one_hot_targets(self.Y_val_raw)
    self.Y_test = self.one_hot_targets(self.Y_test_raw)

    return

  def plot_data_set(self, X_sorted, label_markes={0:"o", 1:"s"}, run=None, window_title_suffix='-'):
    ''' Plots the given sorted data set'''
    
    from matplotlib import pyplot as plt
    from matplotlib import patches

    fig = plt.gcf()
    fig.canvas.set_window_title("XOR data wrapper: {}".format(window_title_suffix))

    for i, Xi in enumerate(X_sorted):
      if Xi.shape == (0,): continue
      plt.scatter(Xi[:,0],Xi[:,1], marker = label_markes[i], label = str(i))
          
    if run is not None:
      plt.title(run)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    
    return
  def plot_training_data_set(self, run=''):
    ''' Plots the current training data set'''
    X_sorted = self.sort_data_set(self.X_train, self.Y_train_raw)
    return self.plot_data_set(X_sorted, run = run, window_title_suffix = "training data")
  def plot_test_data_set(self, run):
    ''' Plots the correct labels for the test data set'''
    X_sorted = self.sort_data_set(self.X_test, self.Y_test_raw)
    return self.plot_data_set(X_sorted, run = run, window_title_suffix = "test data")
  def plot_classifier(self, run, classifier):
    ''' Plots the labels given by the classifier for the test data set'''
   
    predictions = np.apply_along_axis(classifier, 1, self.X_test).argmax(axis = 1)
    X_sorted = self.sort_data_set(self.X_test, predictions)

    return self.plot_data_set(X_sorted, run = run, window_title_suffix = "Classifier Output")