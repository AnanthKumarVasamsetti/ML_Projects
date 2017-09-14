from numpy import loadtxt, zeros, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour


def compute_cost(theta,X,Y,no_of_examples):
    
    hypo = X.dot(theta).flatten()
    sqErrors = (hypo-Y) ** 2
    cost = (1.0/(2*no_of_examples)) * sqErrors.sum()

    return cost

def gradient_descent(X,y,theta,no_of_examples,alpha,num_iters):
    m = y.size
    J_history = zeros(shape=(num_iters, 1))

    for i in range(num_iters):

        # predictions = X.dot(theta).flatten()

        #errors_x1 = (predictions - y) * X[:, 0]
        #errors_x2 = (predictions - y) * X[:, 1]
        
        for j in range(theta.size):
            predictions = X.dot(theta).flatten()
            errors_x1 = (predictions - y) * X[:, j]
            theta[j][0] = theta[j][0] - alpha * (1.0 / m) * errors_x1.sum()
            #theta[1][0] = theta[1][0] - alpha * (1.0 / m) * errors_x2.sum()

        J_history[i, 0] = compute_cost(theta,X, y,m)

    return theta, J_history
    # J_history = zeros(shape=(num_iters,1))

    # for i in range(num_iters):
    #     predictions = X.dot(theta).flatten()
    #     errors_X0 = [predictions - Y] * X[:, 0]
    #     errors_X1 = [predictions - Y] * X[:, 1]

    #     theta[0][0] = theta[0][0] - alpha * (1.0/no_of_examples) *errors_X0.sum()
    #     theta[1][0] = theta[1][0] - alpha * (1.0/no_of_examples) *errors_X1.sum()

    #     J_history[i, 0] = compute_cost(theta,X,Y,no_of_examples)

    #     return theta, J_history

def calculate_hypothesis(data):
    
    X = data[:, 0]
    Y = data[:, 1]
    
    no_of_examples = Y.size

    it = ones(shape=(no_of_examples,2))
    it[:,1] = X

    s = list(it.shape)[1] #it.shape is tuple, it is immutable
    theta = zeros(shape=(s,1))
    iterations = 1500
    alpha = 0.01

    theta, J = gradient_descent(it, Y, theta, no_of_examples, alpha, iterations)
    print("------------------ theta ---------------------")
    print(theta)
    print("------------------- J -------------------")
    print(min(J))

if __name__ == "__main__":
    #Load the dataset
    data = loadtxt('ex1data1.txt', delimiter=',')

    #Plot the data
    scatter(data[:, 0], data[:, 1], marker='x', c='b')
    title('Profits distribution')
    xlabel('Population of City in 10,000s')
    ylabel('Profit in $10,000s')
    #show()

    calculate_hypothesis(data)
