import numpy as np
import random
from nn_funcs import array_tweak
from nn_funcs import cal_size
from activations import relu, sigmoid, tanh, leaky_relu, elu_array

class NN():
    def __init__(self, shape, weights=None, biases=None):
        """
        shape = (n_layer1, n_layer2 ... n_layern)
        """
        if not weights and not biases:
            self.weights = [np.random.uniform(-1,1,(shape[i-1], shape[i])) for i in range(1,len(shape))]
            self.biases = [np.random.uniform(-1,1,(shape[i])) for i in range(1, len(shape))]
        else:
            self.weights = weights
            self.biases = biases

        self.score = 0
        self.shape = shape
        self.size = cal_size(self)

    def mutate(self):

        for i,w_layer in enumerate(self.weights):
            self.weights[i] = array_tweak(w_layer)

        for i,b_layer in enumerate(self.biases):
            self.biases[i] = array_tweak(b_layer)

    def evaluate(self, inputs):

        for i in range(len(self.weights)):
            inputs = np.dot(inputs, self.weights[i])+self.biases[i]
            if i < len(self.weights)-1:
                inputs = relu(inputs)
            # else:
            #     inputs = tanh(inputs)

        return inputs

    def evaluate_bunch(self, bunch_inputs):
        
        output = np.apply_along_axis(self.evaluate, 1, bunch_inputs)
        return output
