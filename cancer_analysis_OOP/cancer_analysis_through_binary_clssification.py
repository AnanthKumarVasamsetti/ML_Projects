from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
from My_Logistic_regression import Logistic_regression

def main():
    cancer_data = datasets.load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(cancer_data.data, cancer_data.target, random_state = 0)

    features = X_train
    features = features.T
    
    labels = y_train.reshape((1, len(y_train)))
    intercept = 0

    #Weights for each and every feature in the dataset
    weights = np.zeros((features.shape[0], 1))
    
    regression = Logistic_regression()
    weights , intercept = regression.calculate_gradient_descent(weights,features,labels,intercept)
    X_test = X_test.T
    prediction = regression.predict(X_test, weights, intercept)
    
    y_test_count = y_test.shape[0]
    y_test = y_test.reshape((1,y_test_count))
    
    regression.calculate_accuaracy(prediction,y_test)

if __name__ == '__main__':
    main()
