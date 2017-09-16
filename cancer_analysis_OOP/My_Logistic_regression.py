import numpy as np
class Logistic_regression:
        
    def calculate_sigmoid(self, z):
        return (1/(1+np.exp(-z)))

    def calculate_AminusY(self, weights, features, labels, intercept):
        z = np.dot(features, weights.T) + intercept
        resultant_matrix = self.calculate_sigmoid(z)
        return (resultant_matrix - labels)
        
    def calculate_gradient_descent(self, weights, features, labels, intercept):
        learning_rate = 0.033
        number_of_iterations = 10000
        m = features.shape[0]

        for i in range(number_of_iterations):
            result = self.calculate_AminusY(weights, features, labels, intercept)
            dw = (np.dot(result.T , features))/m
            weights = weights - (learning_rate * dw)
            intercept = intercept - (learning_rate * (np.sum(result))/m)
        
        weights = np.around(weights)
        return weights, intercept
    
    def predict(self,test_features, weights, intercept):
        z = np.dot(test_features, weights.T) + intercept
        resultant_matrix = self.calculate_sigmoid(z)
        return resultant_matrix
        