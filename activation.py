import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def relu(x):
    x = [max(0,i) for i in x]
    return np.array(x, dtype=float)

def leaky_relu(x, alpha=0.1):
    return np.where(x >= 0, x, alpha*x)