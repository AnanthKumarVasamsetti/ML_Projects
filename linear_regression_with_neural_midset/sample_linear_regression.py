import numpy as np

"""
    This function calculates the sigmoid

    Argument:
    z -- A scalar or numpy array of any size

    Returns:
    s -- Calculated sigmoid value
"""
def sigmoid(z):
    s = 1./(1+np.exp(-z))
    return s

"""
    This function initializes a vector(w) with zeros with dimensions (dim, 1)
    and intercpet(b) to 0

    Argument:
    dim -- size of the vector w that we want (or parameters in this case)

    Returns:
    w -- A vector with dimensions (dim, 1)
    b -- Initialized scalar (corresponds to bias)
"""
def initialize_with_zeros(dim):
    w = np.zeros(shape = (dim, 1), dtype = np.float32)
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b

"""
    Calculates the cost function and gradient descent

    Arguments:
    w -- Weights, a numpy array of size (dim, 1)
    b -- Bias, a scalar
    X -- Data
    Y -- Labels vector

    Returns:
    cost -- Likelihood cost for logistic regression
    grad -- Contains two varaibles "dw" and "db"
            dw : Gradient of the loss with respect to w, thus same shape as w
            db : Gradient of the loss with respect to b, thus same shape as b
"""
def propagate(w, b, X, Y):
    m = X.shape[1] #number of examples

    #Forward propagation (from X to cost)
    A = np.dot(w.T, X) + b
    A = sigmoid(A)  #Compute activation
    cost = (-1. / m) * np.sum((Y * np.log(A) + (1 - Y) * np.log(1 - A)), axis = 1) #Compute cost

    #Backward propagation (to find grad)
    dw = (1./m) * np.dot(X, ((A - Y).T))
    db = (1./m) * np.sum(A - Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw" : dw,
             "db" : db}

    return grads, cost

"""
    This function optimizes w and b by running gradient descent algorithm

    Arguments:
    w -- Weigths, a numpy array
    b -- Bias, a scalar
    X -- Data
    Y -- Labels matrix
    num_iterations -- Number of iterations to run for optimizing w  and b
    learning_rate -- Learning rate of the gradient descent update rule
    print_cost -- True to print cost after every 100 iterations

    Returns:
    params -- A dictionary containing weights w and bias b
    grads -- A dictionary containing dw and db calculated with respect to cost
    cost -- An array containing cost attained after every 100 iterations
"""
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost):
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w = w, b = b, X = X, Y = Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        #Record costs for every 100 iterations
        if(i % 100 == 0):
            costs.append(cost)

        #Show the cost if print_cost is true
        if((print_cost == True) and (i % 100 == 0)):
            print("Cost at iteration ",i,":",cost)

        params = {"w" : w,
                 "b" : b}
        grads = {"dw" : dw,
                 "db" : db}

    return params, grads, costs
"""
    This function predicts output as 0/1 using learned logistic regression using (w, b)

    Arguments:
    w -- Weights, a numpy array
    b -- Bias, a scalar value
    X -- Input data

    Returns:
    Y_predictions -- A numpy array containing predictions for the given input data
"""
def predict(w, b, X):
    m = X.shape[1]
    w = w.reshape((X.shape[0], 1))
    Y_predictions = np.zeros((1, m))

    A = sigmoid((np.dot(w.T, X) + b))

    #[print(x) for x in A]

    for i in range(A.shape[1]):
        if(A[0, i]) >= 0.5:
            Y_predictions[0, i] = 1
        else:
            Y_predictions[0, i] = 0

    assert(Y_predictions.shape == (1, m))

    return Y_predictions

w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
print ("predictions = " + str(predict(params["w"], params["b"], X)))
