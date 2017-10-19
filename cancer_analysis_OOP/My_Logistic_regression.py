import numpy as np

class Logistic_regression:
        
    def calculate_sigmoid(self, z):
        return (1/(1+np.exp(-z)))

    def calculate_AminusY(self, weights, features, labels, intercept):
        z = np.dot(weights.T, features) + intercept
        resultant_matrix = self.calculate_sigmoid(z)
        return (resultant_matrix - labels)
        
    def calculate_gradient_descent(self, weights, features, labels, intercept):
        learning_rate = 1
        number_of_iterations = 10000
        m = features.shape[1]

        for i in range(number_of_iterations):
            result = self.calculate_AminusY(weights, features, labels, intercept)
            dw = (np.dot(features, result.T))/m
            weights = weights - (learning_rate * dw)
            intercept = intercept - (learning_rate * (np.sum(result))/m)
        
        weights = np.around(weights)
        return weights, intercept
    
    def predict(self,test_features, weights, intercept):
        z = np.dot(weights.T, test_features) + intercept
        resultant_matrix = self.calculate_sigmoid(z)
        return np.round(resultant_matrix)

    def calculate_accuaracy(self, output, expected):
        print("================================================================")
        
        errors = 0
        correct = 0
        for i in range(output.shape[1]):
            if output[0][i] != expected[0][i]:
                errors = errors + 1
            else:
                correct = correct + 1

        print("correct :",correct)        
        print("errors :",errors)
        confidence = (correct/output.shape[1]) * 100
        print("Accuracy :",round(confidence,2))

        print("================================================================")