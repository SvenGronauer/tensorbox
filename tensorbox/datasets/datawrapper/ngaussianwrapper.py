from .basewrapper import *

class NGaussianWrapper(BaseWrapper):
  def __init__(self, T=1000, N=6, r=4.0, sigma=0.75, use_file=False, force_use_file=False, *args, **kwargs):
    '''
    Creates a N-Gaussian Wrapper instance.

    If use_file is true, the wrapper first tries to load existing data from the disk. 
    If this fails a new data set is created and stored.

    Parameters
    ----------

    T :         int
                The number of samples in the training set.

    N :         int
                Defines the number of normal distrubtions equadistantly placed on the circle

    r :         float
                The radius of the circle

    sigma :     float
                The standart deviation of the normal distributions

    use_file :  bool 
                If true the training data is restored from file. Allows for reproducible experiments 
                without worrying about setting the seed of numpy

    force_use_file :  bool
                      Enforces the loading of data from file, i.e. IOErrors are not catched.
    '''

    assert all(map(lambda x: x > 0, [T, N, r, sigma])), "The parameters must be positive"

    self.N = int(N)
    self.r = float(r)
    self.sigma = float(sigma)

    # Call base class to create everything
    return super(NGaussianWrapper, self).__init__(T, "NGaussiansDataSet.pickle", use_file = use_file, force_use_file = force_use_file, *args, **kwargs)

  def _create_samples_and_label_them(self, T):
    ''' Call this function to create T samples and their labels '''
    
    dataX = [None for _ in xrange(self.N)]
    dataY = [None for _ in xrange(self.N)]

    theta_step = 2.0 * np.pi / self.N
    thetas = [i * theta_step for i in xrange(self.N)]

    for i,theta in enumerate(thetas):

      Ti = self.T / self.N

      if i == 0:
        Ti += self.T - Ti * self.N

      cx = self.r * np.cos(theta)
      cy = self.r * np.sin(theta)

      Xi = np.random.normal(0.0, self.sigma, size = (Ti,2)).astype(self.dtype)

      Xi[:,0] += cx
      Xi[:,1] += cy

      Yi = np.zeros((Ti,1), dtype = np.int32) + i

      dataX[i] = Xi
      dataY[i] = Yi
     
    #end for
    
    dataX = np.vstack(dataX)
    dataY = np.vstack(dataY)

    self.original_std_dev = dataX.std()

    dataX /= self.original_std_dev

    return dataX, dataY
  def _create_data_set(self, T):
    ''' Implementation of the base class, here the actual data is created '''

    self.X_train, self.Y_train_raw = self._create_samples_and_label_them(self.T)
    self.X_val, self.Y_val_raw = self._create_samples_and_label_them(self.T / 2)
    self.X_test, self.Y_test_raw = self._create_samples_and_label_them(self.T / 2)

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
  
  def plot_data_set(self, X_sorted, label_markes={0:"o", 1:"s", 2:"h", 3:"1", 4:"+",5:"x"}, hide_legend=False, run=None, window_title_suffix='-'):
    ''' Plots the given sorted data set'''
    
    from matplotlib import pyplot as plt
    from matplotlib import patches

    fig = plt.gcf()
    fig.canvas.set_window_title("N Gaussian data wrapper: {}".format(window_title_suffix))

    for i, Xi in enumerate(X_sorted):
      if Xi.shape == (0,): continue
      #plt.scatter(Xi[:,0],Xi[:,1], marker = label_markes[i], label = str(i))
      plt.scatter(Xi[:,0],Xi[:,1],15, label = str(i),                   
                  edgecolors = 'k',
                  linewidth = 1)

    ax = fig.gca()
    ax.set_axisbelow(True)
    ax.add_patch(patches.Circle((0,0), self.r / self.original_std_dev,fill = False))

    if run is not None:
      plt.title(run)

    largest_val = max(map(lambda x: np.abs(x).max(), X_sorted))

    border = 1.05 * max(self.r, largest_val)

    plt.xlim((-border, +border))
    plt.ylim((-border, +border))
    plt.xlabel("x")
    plt.ylabel("y")
    if not hide_legend:
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
   
    predictions = np.apply_along_axis(classifier, 1, self.X_test).argmax(axis = 1).reshape(-1,1)
    X_sorted = self.sort_data_set(self.X_test, predictions)

    return self.plot_data_set(X_sorted, run = run, window_title_suffix = "Classifier Output")