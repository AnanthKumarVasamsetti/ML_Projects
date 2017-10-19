from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def calculate_sigmoid(z):
    return (1/(1+np.exp(-z)))

def calculate_AminusY(weights, features, labels, intercept):
    
    z = np.dot(weights.T,features) + intercept
    #print(features.shape)
    resultant_matrix = calculate_sigmoid(z)
    return (resultant_matrix - labels)

def calculate_gradient_descent(weights, features, labels, intercept):
    learning_rate = 0.033
    number_of_iterations = 10000
    m = features.shape[0]

    for i in range(number_of_iterations):
        result = calculate_AminusY(weights, features, labels, intercept)
        dw = (np.dot(features, result.T))/m
        weights = weights - (learning_rate * dw)
        intercept = intercept - (learning_rate * (np.sum(result))/m)

    weights = np.around(weights)
    return weights, intercept

def main():
    digits = datasets.load_breast_cancer()

    features = np.array(digits.data, 'int16')
    labels = np.array(digits.target,'int').reshape((1,569))

    features = features.T
    #labels = labels.T
    intercept = 0
    weights = np.zeros((features.shape[0],1))
    weights, intercept = calculate_gradient_descent(weights, features, labels, intercept)

    result = np.dot(weights.T, features) + intercept
    result = calculate_sigmoid(result)
    result = np.around(result)

    errors = 0
    correct = 0
    for i in range(result.shape[1]):
        if result[0][i] != labels[0][i]:
            errors = errors + 1
        else:
            correct = correct + 1

    print("errors :",errors)
    print("correct :",correct)
    confidence = (correct/result.shape[1]) * 100
    print("Accuracy :",round(confidence,2))
    
if __name__ == '__main__':
    main()
