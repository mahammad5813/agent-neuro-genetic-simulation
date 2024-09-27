import numpy as np
import math
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0,x)

def elu(x):
    if x > 0 :
        return x
    else : 
        return (np.exp(x)-1)
    
elu_array = np.vectorize(elu)
    
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def leaky_relu(x):
    return np.maximum(0.1*x, x)

if __name__ == "__main__":
    x = np.arange(-10,10,0.1)
    y = [leaky_relu(i) for i in x]
    plt.plot(x,y)
    plt.show()