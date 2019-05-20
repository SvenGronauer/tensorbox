from .basewrapper import *

class RhodoneaWrapper(BaseWrapper):
  ''' 
  A regression dataset based on parameterized rhodonea (or rose) curves. 
  
  The goal is to map a line to an arbitrary rose.
  '''

  def __init__(self, use_file=False, T=1000, force_use_file=False, recreate=False, add_noise=True, *args, **kwargs):
    '''
    Creates a Rhodonea Wrapper instance.

    If use_file is true, the wrapper first tries to load existing data from the disk. 
    If this fails a new data set is created and stored.

    => use_file = True allows for reproducible experiments without worrying about setting the seed of numpy
    '''

    # Parameters to specify the data set
    # Values are chosen based on the table: https://en.wikipedia.org/wiki/Rose_(mathematics)

    # Amplidutes are simple scalaing factors for the whole rose
    self.amplitude = 5.0
    
    # The two parameters of rhodonea curves
    self.n = 3
    self.d = 7
       
    # Scalar factor in front of the curve parameter t (only for the radius)
    self.k = 1.0 * self.n / self.d
        
    # Whether to add noise to the curve
    self.add_noise = add_noise

    # Input sample Std deviation for gaussian noise, currently 1.25% of amplitde
    self.stdX = 0.5 * 0.025 * self.amplitude

    # Label Std deviation for gaussian noise, currently 2.5% of amplitde
    self.stdY = 0.025 * self.amplitude

    # Call base class to create everything
    return super(RhodoneaWrapper, self).__init__(T, "RhodoneaDataSet.pickle", use_file = use_file, force_use_file = force_use_file, recreate = recreate, *args, **kwargs)
 
  def _rhodonea_curve(self, t):
    ''' Evaluates the lissajous curves '''
    r = self.amplitude * np.cos(self.k * t)
    p = np.array([np.cos(t),np.sin(t)]).T

    # radii r (T,) is vector, points p (T,2) is matrix -> this is row-wise multiplication with scalar  
    return np.multiply(r[:,np.newaxis], p)

  def _create_curve(self, T):
    ''' Call this function to create T samples and their labels '''
    
    p = 2 if (self.n * self.d % 2) == 0 else 1

    t = np.linspace(0.0, np.pi * self.d * p, T, dtype = self.dtype)

    r = np.linspace(1.0, 3.0, T, dtype = self.dtype)

    X = np.multiply(r[:,np.newaxis], np.array([np.cos(t), np.sin(t)]).T  )
    Y = self._rhodonea_curve(t)

    if self.add_noise:
      X+= np.random.normal(0.0, self.stdX, size = X.shape)
      Y+= np.random.normal(0.0, self.stdY, size = Y.shape)

    return X,Y
  def _create_data_set(self, T):
    ''' Implementation of the base class, here the actual data is created '''

    self.X_train, self.Y_train = self._create_curve(self.T)
    self.X_val, self.Y_val = self._create_curve(self.T / 2)
    self.X_test, self.Y_test = self._create_curve(self.T / 2)

    self.Y_train_raw = None
    self.Y_val_raw = None
    self.Y_test_raw = None

    return

  def _serialize(self):
    ''' Shall return what needs to be stored in the pickle file '''
    return self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test
  def _deserialize(self, result):
    ''' Gets the result of loading a pickle file and shall parse that information and update self '''

    self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test = result
    
    self.Y_train_raw = None
    self.Y_val_raw = None
    self.Y_test_raw = None

    return

  def plot_data_set(self, X, Y, run=None, window_title_suffix='-'):
    ''' Plots the given sorted data set'''
    
    from matplotlib import pyplot as plt
    from matplotlib import patches

    fig = plt.gcf()
    
    fig.canvas.set_window_title("Rhodonea data wrapper: {}".format(window_title_suffix))

    t = np.linspace(0,1, X.shape[0])

    fig.add_subplot(121)
    plt.plot(X[:,0],X[:,1], '-k')
    plt.scatter(X[:,0], X[:,1], c = t, cmap = 'hsv')

    if run is not None:
      plt.title(run)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.grid()
    
    fig.add_subplot(122)
    plt.plot(Y[:,0],Y[:,1], '-k')
    plt.scatter(Y[:,0],Y[:,1], c = t, cmap = 'hsv')

    if run is not None:
      plt.title(run)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.grid()

    #ax = fig.gca()
    #ax.set_axisbelow(True)
    #ax.add_patch(patches.Rectangle((self.x_range[0],self.y_range[0]), self.x_range[1] - self.x_range[0], self.y_range[1] - self.y_range[0],fill = False))
    #ax.add_patch(patches.Circle((0,0), self.radii[1],fill = False))
    #ax.add_patch(patches.Circle((0,0), self.radii[2],fill = False))
    #ax.add_patch(patches.Arc((0,0), 2.0 * self.radii[0],2.0 * self.radii[0],90.0,180.0,fill = False))
    #ax.add_patch(patches.Polygon(np.array([(0.0,self.radii[2]),(0.0,self.radii[0])]),closed = True,edgecolor = 'k'))
    #ax.add_patch(patches.Polygon(np.array([(0.0,-self.radii[0]),(0.0,-self.radii[2])]),closed = True,edgecolor = 'k'))
    #ax.add_patch(patches.Polygon(np.array([(self.x_range[0],0.0),(-self.radii[2],0.0)]),closed = True,edgecolor = 'k'))
    #ax.add_patch(patches.Polygon(np.array([(self.x_range[1],0.0),(self.radii[2],0.0)]),closed = True,edgecolor = 'k'))
    
    return
  def plot_training_data_set(self, run=''):
    ''' Plots the current training data set. '''
    return self.plot_data_set(self.X_train, self.Y_train, run = run, window_title_suffix = "training data")
  def plot_test_data_set(self, run=''):
    ''' Plots the current training data set. '''
    return self.plot_data_set(self.X_test, self.Y_test, run = run, window_title_suffix = "test data")  
  def plot_regressor(self, run, regressor, batched=False, already_predicted=False):
    ''' Plots the output of the regressor. '''
   
    if already_predicted:
      regressions = regressor
    else:
      if batched:
        regressions = regressor(self.X_test)
      else:
        regressions = np.apply_along_axis(regressor, 1, self.X_test)
      #end
    #end

    return self.plot_data_set(self.X_test, regressions, run = run, window_title_suffix = "Regressor Output")
