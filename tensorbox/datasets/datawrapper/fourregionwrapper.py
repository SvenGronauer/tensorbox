from .basewrapper import *


class FourRegionWrapper(BaseWrapper):

  def __init__(self, use_file=False, T=1000, force_use_file=False, recreate=False, *args, **kwargs):
    '''
    Creates a FourRegion Wrapper instance.

    If use_file is true, the wrapper first tries to load existing data from the disk. 
    If this fails a new data set is created and stored.

    => use_file = True allows for reproducible experiments without worrying about setting the seed of numpy
    '''

    # Parameters to specify the data set
    self.x_range = (-4,4)
    self.y_range = (-4,4)
    self.radii = [1.0,2.0,3.0]

    # Call base class to create everything
    return super(FourRegionWrapper, self).__init__(T, "FourRegionDataSet.pickle", use_file = use_file, force_use_file = force_use_file, recreate = recreate, *args, **kwargs)
  
  def _region_1(self, x):
    ''' Returns true, if the sample x lies within this region '''
    return x[1] > 0.0 and np.linalg.norm(x) > self.radii[2]
  def _region_2(self, x):
    ''' Returns true, if the sample x lies within this region '''
    return x[1] <= 0.0 and np.linalg.norm(x) > self.radii[2]
  def _region_3(self, x):
    ''' Returns true, if the sample x lies within this region '''
    r = np.linalg.norm(x)

    if x[0] <= 0.0:
      return self.radii[1] <= r <= self.radii[2]
    else:
      return self.radii[0] <= r <= self.radii[1]
  def _region_4(self, x):
    ''' Returns true, if the sample x lies within this region '''
    r = np.linalg.norm(x)

    if x[0] <= 0.0:
      return 0.0 <= r <= self.radii[1]
    else:
      return 0.0 <= r <= self.radii[0] or self.radii[1] <= r <= self.radii[2]
  def _get_label(self, x):
    ''' Returns the label for the sample x by checking the regions '''
    if self._region_1(x):
      return 0
    elif self._region_2(x):
      return 1
    elif self._region_3(x):
      return 2
    elif self._region_4(x):
      return 3
    else:
      raise ValueError("The sample {} can not be labeld".format(x))
  def _create_samples_and_label_them(self, T):
    ''' Call this function to create T samples and their labels '''
    
    X = np.array([np.random.uniform(*self.x_range, size = (T,)),
                  np.random.uniform(*self.y_range, size = (T,))], dtype = self.dtype).T

    Y = np.array(map(self._get_label, X)).reshape(-1,1)

    return X,Y

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

    return True

  def plot_data_set(self, X_sorted, label_markes={0:"o", 1:"s", 2:"h", 3:"1"}, run = None, window_title_suffix = '-'):
    ''' Plots the given sorted data set'''
    
    from matplotlib import pyplot as plt
    from matplotlib import patches

    fig = plt.gcf()    
    fig.canvas.set_window_title("Four Region data wrapper: {}".format(window_title_suffix))

    for i, Xi in enumerate(X_sorted):
      if Xi.shape == (0,): continue
      plt.scatter(Xi[:,0],Xi[:,1], marker = label_markes[i], label = str(i))

    ax = fig.gca()
    ax.set_axisbelow(True)
    ax.add_patch(patches.Rectangle((self.x_range[0],self.y_range[0]), self.x_range[1] - self.x_range[0], self.y_range[1] - self.y_range[0],fill = False))
    ax.add_patch(patches.Circle((0,0), self.radii[1],fill = False))
    ax.add_patch(patches.Circle((0,0), self.radii[2],fill = False))
    ax.add_patch(patches.Arc((0,0), 2.0 * self.radii[0],2.0 * self.radii[0],90.0,180.0,fill = False))
    ax.add_patch(patches.Polygon(np.array([(0.0,self.radii[2]),(0.0,self.radii[0])]),closed = True,edgecolor = 'k'))
    ax.add_patch(patches.Polygon(np.array([(0.0,-self.radii[0]),(0.0,-self.radii[2])]),closed = True,edgecolor = 'k'))
    ax.add_patch(patches.Polygon(np.array([(self.x_range[0],0.0),(-self.radii[2],0.0)]),closed = True,edgecolor = 'k'))
    ax.add_patch(patches.Polygon(np.array([(self.x_range[1],0.0),(self.radii[2],0.0)]),closed = True,edgecolor = 'k'))

    if run is not None:
      plt.title(run)

    plt.xlabel("x")
    plt.xlim(self.x_range)
    plt.ylim(self.y_range)
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
  def plot_classifier(self, run, classifier, batched=False, already_predicted=False):
    ''' Plots the labels given by the classifier for the test data set'''
   
    if already_predicted:
      predictions = classifier
    else:
      if batched:
        predictions = np.argmax(classifier(self.X_test), axis = 1).reshape(-1,1)

      else:
        predictions = np.apply_along_axis(classifier, 1, self.X_test).argmax(axis = 1).reshape(-1,1)
      #end
    #end

    X_sorted = self.sort_data_set(self.X_test, predictions)

    return self.plot_data_set(X_sorted, run = run, window_title_suffix = "Classifier Output")