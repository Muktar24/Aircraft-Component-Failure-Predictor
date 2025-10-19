import numpy as np

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def gradient(theta, X, y):
    m = y.size
    return (X.T @ (sigmoid(X @ theta) - y)) / m

def gradient_descent(X, y, lr=0.1, iter=1000, tol=1e-10):
    X_b = np.c_[np.ones((X.shape[0],1)), X]
    theta = np.zeros(X_b.shape[1])
    for i in range(iter):
        grad = gradient(theta, X_b, y)
        theta -= lr * grad
        if np.linalg.norm(grad) < tol:
            break
    return theta

def predict_prob(X, theta):
    X_b = np.c_[np.ones((X.shape[0],1)), X]
    return sigmoid(X_b @ theta)

def predict(X, theta, thres=0.5):
    return (predict_prob(X, theta) >= thres).astype(int)



