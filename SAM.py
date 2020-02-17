import numpy as np
from Technical import y
from Technical import expanded_inputs
from Technical import synaptic_weights


class NeuralNetwork:
    def __init__(self, expanded_inputs, y):
        self.input      = expanded_inputs
        self.weights1   = synaptic_weights             
        self.y          = y
        self.OutY       = y
              

    def feedforward(self):

        inputY = self.y

        I = np.identity(len(self.input.T))

        square = []

        lamb = 1
        for i in range(100):

            #Transpose of the inverse matrix ((M^tM)^-1 M^t)^t y            
            d_weights1 = np.linalg.inv(self.input.T.dot(self.input) + lamb * I).dot(self.input.T).dot(inputY)

            self.OutY = self.input.dot(d_weights1)
          
            square.append(np.sum((inputY - self.OutY)**2))

            # print(f'Weights = {d_weights1}')
            # print(f'Square = {square}')

            inputY = self.OutY
        return d_weights1


if __name__ == "__main__":
    
    nn = NeuralNetwork(expanded_inputs,y)

    weights = nn.feedforward()

