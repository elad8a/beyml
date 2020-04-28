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
    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.3, random_state=42)

    # create the model
    model = Sequential()
    # add layers
    model.add(Dense(units=1, input_shape=(2,), activation='sigmoid'))

    model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=100, verbose=1)
    #data = kbe.variable(np.random.random((4, 2)))
    #zeros_data = kbe.zeros_like(data)
    #res = kbe.eval(zeros_data)

    aa, bb = plh.get_meshgrid_2d_learning_set(X)

    ab_combinations = np.array([aa.ravel(),bb.ravel()]).T

    model_predictions = model.predict(ab_combinations).reshape(aa.shape)
    
    evaluation_result = model.evaluate(X_validate, y_validate)

    print('test loss = ', evaluation_result[0], ' accuracy = ', evaluation_result[1])


    # plot countours to show model performance
    plh.plot_filled_contour_around_predictions(plt, aa, bb, model_predictions)  
    
    # plot learning set 
    plh.plot_labeled_features(plt, X, y)

    #plt.figure(figsize=(12, 8))

    plt.show()
        
    #print(res)



main()
