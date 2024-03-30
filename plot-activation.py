import numpy as np
import matplotlib.pyplot as plt
from activation import sigmoid,tanh,relu,leaky_relu

#####Comments for Bug Fix branch

def plot_sigmoid(x):
    y = sigmoid(x)
    plt.plot(x,y)
    plt.xlabel('Input')
    plt.ylabel('Sigmoid Output')
    plt.title('Sigmoid Activation Function')
    plt.grid(True)
    plt.show()

def plot_tanh(x):
    y = tanh(x)
    plt.plot(x,y)
    plt.xlabel('Input')
    plt.ylabel('Tanh Output')
    plt.title('Tanh Activation Function')
    plt.grid(True)
    plt.show()

def plot_relu(x):
    y = relu(x)
    plt.plot(x,y)
    plt.xlabel('Input')
    plt.ylabel('Relu Output')
    plt.title('Relu Activation Function')
    plt.grid(True)
    plt.show()

def plot_leaky_relu(x):
    y = leaky_relu(x)
    plt.plot(x,y)
    plt.xlabel('Input')
    plt.ylabel('Leaky Relu Output')
    plt.title('Leaky Relu Activation Function')
    plt.grid(True)
    plt.show()

random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]
x = np.array(random_values)
#plot_sigmoid(x)
#plot_tanh(x)
#plot_relu(x)
#plot_leaky_relu(x)

print("ReLU: ", relu(x))
print("Leaky ReLU: ", leaky_relu(x))
print("Tanh: ", tanh(x))
print("Bug Fix Branch")
