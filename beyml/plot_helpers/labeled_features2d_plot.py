import numpy as np

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




def get_meshgrid_2d_learning_set(X, margin = 0.1, tick_per_axis = 101):
    '''
    given a 2d feature set X, this function
    returns a meshgrid that represents the 2d space covered by this set
    '''
    # extract the minimal value of each column, 
    # axis=0 means that we find the minimal value when running through the rows.
    # there are 2 columns for each row, so we get 2 values that are the minimus on each column

    amin, bmin = X.min(axis=0) - margin
    amax, bmax = X.max(axis=0) + margin

    horizontal_ticks = np.linspace(amin, amax, tick_per_axis)
    vertical_ticks = np.linspace(bmin, bmax, tick_per_axis)
    
    aa, bb = np.meshgrid(horizontal_ticks, vertical_ticks) # aa.shape = bb.shape = (TICKS_COUNT, TICKS_COUNT)

    # aa runs horizontaly and duplicated verticaly
    # bb runs verticaly and duplicated horizontaly

    return aa, bb


def plot_filled_contour_around_predictions(plt, xx, yy, predictions):
    plt.contourf(xx,yy,predictions, cmap='bwr', alpha=0.2)
    return plt
