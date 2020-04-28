import plaidml.keras
plaidml.keras.install_backend()

import numpy as np
from keras import backend as kbe
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt



from plot_helpers import labeled_features2d_plot as plh




def main():

    # create 2 labeled clusters around 2 centers
    # X will be the 2d feature coordinates
    # y will be the center index which will be used as 0-1 label
    X, y = make_blobs(n_samples=1000, centers=2, random_state=42)

    # split the sets into train & validate
    X_train, X_validate, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # create the model
    model = Sequential()
    # add layers
    model.add(Dense(units=1, activation='sigmoid', input_shape=(2,)))
    #data = kbe.variable(np.random.random((4, 2)))
    #zeros_data = kbe.zeros_like(data)
    #res = kbe.eval(zeros_data)

    plt.figure(figsize=(12, 8))
    plh.plot_labeled_features(plt, X, y)
    plt.show()
        
    #print(res)



main()
