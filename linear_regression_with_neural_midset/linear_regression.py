"""
    This program uses logistic regression to classify a given image
"""
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
import sample_linear_regression as lr

np.set_printoptions(threshold=np.nan)
"""
    This function processes the train data and predicts output for test data

    Arguments:
    X_train -- Features matrix of train data
    Y_train -- Labels matrix of train data
    X_test -- Features matrix of test data
    Y_test -- Labels matrix of test data
    number_of_iterations -- Number of times the optimization loop should run
    learning_rate -- Learning rate of the gradient descent update rule
    print_cost -- To print the cost after every 100 iterations

    Retrun:
    None
"""
def model(X_train, Y_train, X_test, Y_test, number_of_iterations, learning_rate, print_cost):
    m = X_train.shape[1]
    w, b = lr.initialize_with_zeros(X_train.shape[0])

    params, grads, costs = lr.optimize(w, b, X_train, Y_train, number_of_iterations, learning_rate, print_cost)

    Y_predictions_test = lr.predict(params["w"], params["b"], X_test)
    Y_predictions_train = lr.predict(params["w"], params["b"], X_train)

    train_accuracy = 100 - np.mean(np.abs(Y_predictions_train - Y_train)) * 100
    test_accuracy = 100 - np.mean(np.abs(Y_predictions_test - Y_test)) * 100

    print("Training dataset accuracy: ", train_accuracy, " %")
    print("Test dataset accuracy: ", test_accuracy, " %")

"""
    This function loads the dataset and flattens the features matrix for the image

    Arguments:
    None

    Returns:
    None
"""
def initilize():
    # Loading the data (cat/non-cat)
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    #Show dimensions
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]

    print("number of training examples: ",m_train)
    print("number of testing examples: ",m_test)
    print("number of pixels in each image(square): ",num_px)

    #Reshaping training and testing datasets
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    train_set_x = train_set_x_flatten/255
    test_set_x = test_set_x_flatten/255
    model(train_set_x, train_set_y, test_set_x, test_set_y, 2000, 0.3, True)

initilize()
