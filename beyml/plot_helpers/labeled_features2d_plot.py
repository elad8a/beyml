
def plot_labeled_features(plt, X, y):
    '''
    plt - matplotlib.pyplot instance
    X - 2d array object where each row is an instance and each column is a feature
        in this case we have 2 columns that correspond to 2 features
    y - 1d array of 0-1 labels that correspond to each row
    '''

    # plot all instances labeled 0. horizontal axis is feature0 vals, vertical axis is feature1 vals
    
    if (X.shape[0] != y.shape[0] or X.shape[1] != 2):
        raise RuntimeError("incompatible dimensions for plot")
    plt.plot(X[y==0, 0], X[y==0, 1], 'ob', alpha=0.5)
    
    # same way, plot all instances labeled 1
    plt.plot(X[y==1, 0], X[y==1, 1], 'xr', alpha=0.5)

    # set the legends for each previously added plot
    plt.legend(['0', '1'])

    return plt

def plot_decision_boundary(model, X, y):
    
    # extract the minimal value of each column, axis 0 means that we treat the array
    # as a multiaxis table, so that we obtain the min value for each row
    MARGIN = 0.1
    TICKS_COUNT = 101
    amin, amax = X.min(axis=0) - MARGIN
    amax, bmax = X.max(axis=0) - MARGIN

    horizontal_ticks = np.linespace(amin, amax, 101)
    vertical_ticks = np.linspace(bmin, bmax, 101)
    aa, bb = np.meshgrid(horizontal_ticks, vertical_ticks)
    pass


