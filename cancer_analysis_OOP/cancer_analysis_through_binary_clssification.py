from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
from My_Logistic_regression import Logistic_regression

def calculate_success_rate(user_output, expected_output):    
    errors = 0
    correct = 0
    for i in range(len(user_output)):
        if user_output[i][0] != expected_output[i][0]:
            errors = errors + 1
        else:
            correct = correct + 1
    
    print("**********************************************************************")
    print("Total :",len(user_output))
    print("Failure count :",errors)
    print("Success count :",correct)

    err_percentage = (correct/len(user_output)) * 100
    
    print("Error percentage :",round(err_percentage, 5))

    print("**********************************************************************")



def main():
    cancer_data = datasets.load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(cancer_data.data, cancer_data.target, random_state = 0)

    features = X_train
    labels = y_train.reshape((len(y_train),1))
    intercept = 0

    #Weights for each and every feature in the dataset
    weights = np.zeros((1, features.shape[1]))
    
    regression = Logistic_regression()
    weights , intercept = regression.calculate_gradient_descent(weights,features,labels,intercept)

    prediction = regression.predict(X_test, weights, intercept)
    print(prediction)
if __name__ == '__main__':
    main()
