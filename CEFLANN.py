import random
import numpy as np


from Technical import y
from Technical import expanded_inputs
from Technical import I
from Technical import synaptic_weights
from Technical import lamb

# def sigmoid(expanded_inputs):
#     return 1.0/(1+ np.exp(-expanded_inputs))

# def sigmoid_derivative(expanded_inputs):
#     return X * (1.0 - expanded_inputs)

class NeuralNetwork:
    def __init__(self, expanded_inputs, y):
        self.input      = expanded_inputs
        self.weights1   = synaptic_weights             
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        # self.layer1 = sigmoid(np.dot(self.input.T, self.weights1))
        # self.output = sigmoid(np.dot(self.layer1, self.weights2))
        self.output = np.dot(self.input.T, self.weights1)
        

    def Weights(self):

        # application of the chain rule to find derivative of the loss function with respect to weights1  
        d_weights1 = np.linalg.lstsq(self.input.T.dot(self.input) + lamb * I, self.input.T.dot(self.y))[0]
        # d_weights1 = ((self.input.T.dot(self.input) + lamb * I)**-1) * (self.input.T.dot(y))
        # update the weights with the derivative (slope) of the loss function

        self.weights1 += d_weights1


if __name__ == "__main__":
    
    nn = NeuralNetwork(expanded_inputs,y)

    for i in range(1000):
        nn.feedforward()

    print(nn.output)

    output = nn.output