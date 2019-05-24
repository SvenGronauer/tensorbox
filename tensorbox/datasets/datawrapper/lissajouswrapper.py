from .basewrapper import *


class LissajousWrapper(BaseWrapper):
    '''
    A regression dataset based on parameterized lissajous curves.
    The goal is to map a unit circle to an arbitrary Lissajous curve.
    The challenge is that the unit circle and the curves are different
    homeotopic types (depends on parameter choice)
    '''

    def __init__(self,
                 use_file=False,
                 T=1024,
                 force_use_file=False,
                 recreate=False,
                 add_noise=True,
                 *args,
                 **kwargs):
        '''
        Creates a Lissajous Wrapper instance.

        If use_file is true, the wrapper first tries to load existing data from the disk.
        If this fails a new data set is created and stored.

        => use_file = True allows for reproducible experiments without worrying about setting the seed of numpy
        '''

        # Parameters to specify the data set
        # Values are chosen based on the table: https://de.wikipedia.org/wiki/Lissajous-Figur

        # Amplidutes are simple scalaing factors in x and y direction
        self.amplitude = kwargs.get('amplitude', (5.0, 5.0))

        # Scalar factor in front of the curve parameter t
        self.frequency = kwargs.get('frequency', (2.0, 3.0))

        # Scalar bias which is added to the curve parameter t
        self.phase = kwargs.get('phase', (5.0 / 8.0 * np.pi, 0.0))

        # Whether to add noise to the curve
        self.add_noise = add_noise

        # Std deviation for gaussian noise, currently 2.5% of amplitude for labels
        # and half the value for input samples
        self.stdX = kwargs.get('std_x', 0.5 * 0.025 * max(self.amplitude))
        self.stdY = kwargs.get('std_y', 0.025 * max(self.amplitude))

        # Update kwargs with normal args to allow easy base calling
        kwargs.setdefault("T", T)
        kwargs.setdefault("data_name", "LissajousDataSet.pickle")
        kwargs.setdefault("use_file",use_file)
        kwargs.setdefault("force_use_file", force_use_file)
        kwargs.setdefault("recreate", recreate)

        # Call base class to create everything
        super(LissajousWrapper, self).__init__(*args, **kwargs)

    def _lissajous_curve(self, t):
        ''' Evaluates the lissajous curves '''
        return np.array([self.amplitude[0] * np.sin(self.frequency[0] * t + self.phase[0]),
                         self.amplitude[1] * np.sin(self.frequency[1] * t + self.phase[1])])

    def _create_curve(self, T, allow_noise=False):
        ''' Call this function to create T samples and their labels '''

        t = np.linspace(0.0, 2.0 * np.pi, T, dtype = self.dtype)

        X = np.array([np.cos(t), np.sin(t)]).T
        Y = self._lissajous_curve(t).T

        if self.add_noise:
            X += np.random.normal(0.0, self.stdX, size = X.shape)
            Y += np.random.normal(0.0, self.stdY, size = Y.shape)

        if allow_noise:
            X += np.random.normal(0.0, self.stdX, size=X.shape)
            Y += np.random.normal(0.0, self.stdY, size=Y.shape)

        return X, Y

    def _create_data_set(self, T):
        ''' Implementation of the base class, here the actual data is created '''

        self.X_train, self.Y_train = self._create_curve(self.T, allow_noise=True)
        self.X_val, self.Y_val = self._create_curve(self.T / 2, allow_noise=True)
        self.X_test, self.Y_test = self._create_curve(self.T * 2, allow_noise=False)

        self.Y_train_raw = None
        self.Y_val_raw = None
        self.Y_test_raw = None

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

        fig.canvas.set_window_title("Lissajou data wrapper: {}".format(window_title_suffix))

        t = np.linspace(0, 1, X.shape[0])

        fig.add_subplot(121)
        plt.plot(X[:, 0], X[:, 1], '-k')
        # plt.scatter(X[:,0], X[:,1], c = t, cmap = 'hsv')
        plt.scatter(X[:, 0], X[:, 1], c=t, cmap='hsv')

        if run is not None:
            plt.title(run)

        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar()
        plt.grid()

        fig.add_subplot(122)
        plt.plot(Y[:, 0], Y[:, 1], '-k')
        plt.scatter(Y[:, 0], Y[:, 1], c=t, cmap='hsv')

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

        # return
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


class InfinityWrapper(LissajousWrapper):
    ''' A regression dataset based on a curve representing the infinity symbol oo. See the Lissajous figure for details. '''

    def __init__(self, use_file=False, T=1000, force_use_file=False, recreate=False, add_noise=True, *args, **kwargs):
        '''
        Creates a Lissajous Wrapper instance.

        If use_file is true, the wrapper first tries to load existing data from the disk.
        If this fails a new data set is created and stored.

        => use_file = True allows for reproducible experiments without worrying about setting the seed of numpy
        '''

        # Parameters to specify the data set
        # Values are chosen based on the table: https://de.wikipedia.org/wiki/Lissajous-Figur

        # Amplidutes are simple scalaing factors in x and y direction
        kwargs.setdefault('amplitude', (5.0,5.0))

        # Scalar factor in front of the curve parameter t
        kwargs.setdefault('frequency', (1.0,2.0))

        # Scalar bias which is added to the curve parameter t
        kwargs.setdefault('phase', (0.0, 0.0))

        # Whether to add noise to the curve
        kwargs.setdefault('add_noise', add_noise)

        # Std deviation for gaussian noise, currently 2.5% of amplitude
        kwargs.setdefault('std', 0.025 * max(kwargs.get('amplitude')))

        # Update kwargs with normal args to allow easy base calling
        kwargs.setdefault("T",T)
        kwargs.setdefault("data_name", "InfinityDataSet.pickle")
        kwargs.setdefault("use_file",use_file)
        kwargs.setdefault("force_use_file", force_use_file)
        kwargs.setdefault("recreate", recreate)

        # Call base class to create everything
        return super(InfinityWrapper, self).__init__(*args, **kwargs)