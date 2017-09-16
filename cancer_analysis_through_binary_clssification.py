from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def calculate_sigmoid(z):
    return (1/(1+np.exp(-z)))

def calculate_AminusY(weights, features, labels, intercept):
    z = np.dot(features, weights.T) + intercept
    #print(features.shape)
    resultant_matrix = calculate_sigmoid(z)
    return (resultant_matrix - labels)
    
def calculate_gradient_descent(weights, features, labels, intercept):
    learning_rate = 0.033
    number_of_iterations = 10000
    m = features.shape[0]

    for i in range(number_of_iterations):
        result = calculate_AminusY(weights, features, labels, intercept)        
        dw = (np.dot(result.T , features))/m
        weights = weights - (learning_rate * dw)
        intercept = intercept - (learning_rate * (np.sum(result))/m)
    
    weights = np.around(weights)
    return weights, intercept

def main():
    digits = datasets.load_breast_cancer()

    features = np.array(digits.data, 'int16')
    labels = np.array(digits.target,'int').reshape((569,1))

    #features = features.T
    #labels = labels.T
    intercept = 0

    weights = np.zeros((1,features.shape[1]))
    weights, intercept = calculate_gradient_descent(weights, features, labels, intercept)
    
    result = np.dot(features, weights.T) + intercept
    result = calculate_sigmoid(result)
    result = np.around(result)
    
    errors = 0
    correct = 0
    for i in range(len(result)):
        if result[i][0] != labels[i][0]:
            errors = errors + 1
        else:
            correct = correct + 1
    
    print("errors :",errors)
    print("correct :",correct)
    err_percentage = (correct/len(result)) * 100
    print("error percentage :",err_percentage)
    print(intercept)
if __name__ == '__main__':
    main()
