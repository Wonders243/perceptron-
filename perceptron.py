import numpy as np

def initialize(X) :
    W,b =  np.random.randn(X.shape[1],1), np.random.randn(1) 
    return W, b

def model(X, W, b) :
    Y_predict= X.dot(W) + b # droite de regression
    A = 1/(1+ np.exp(-Y_predict)) # fonction d'activation sigmoide
    return A

def cost_function(y, a) :
    m = y.shape[0]
    epsilon = 1e-15
    a = np.clip(a, epsilon, 1 - epsilon)

    cost = - (1/m) * np.sum(y * np.log(a) + (1-y) * np.log(1-a))
    return cost

def gradient_descent(X, y, y_predict, W, b, learning_rate) :
    m = X.shape[0]
    y = y.reshape(-1, 1)
    
    dW = (1/m) * np.dot(X.T, (y_predict - y))
    db = (1/m) * np.sum(y_predict - y)

    W -= learning_rate * dW
    b -= learning_rate * db
    
    return W, b

def train(X, y, learning_rate=0.001, epochs=10) :
    W, b = initialize(X)
    w_list = []
    b_list = []
    cost= []
    y = y.reshape(-1, 1)
    for epoch in range(epochs) :
        y_predict = model(X, W, b)
        cost.append( cost_function(y, y_predict))
        W, b = gradient_descent(X, y, y_predict, W, b, learning_rate)
        w_list.append(W)
        b_list.append(b)

    return w_list, b_list , cost

def predict(X, W, b, threshold=0.5) :
    y_predict = model(X, W, b)
    y_classes = (y_predict >= threshold).astype(int)
    return y_classes


