import numpy as np
from math import exp

class Layer():
    # Neural Network Layer interface
    def __init__(self):
        pass

    def forward(self, input):
        pass

    def isWeighted(self):
        pass

    def getDimensions(self):
        pass

class Dense(Layer):
    # Fully connected layer
    def __init__(self, input_units, output_units):
        self.input_units, self.output_units = input_units, output_units
        self.weights = np.random.randn(input_units, output_units)
        self.bias = np.zeros((1, output_units))

    def forward(self, input):
        return np.dot(input, self.weights) + self.bias

    def isWeighted(self):
        return True

    def getWeights(self):
        return self.weights

    def setWeights(self, weights):
        assert(weights.shape == self.weights.shape)
        self.weights = weights

    def getBias(self):
        return self.bias

    def setBias(self, bias):
        assert(bias.shape == self.bias.shape)
        self.bias = bias

    def getDimensions(self):
        return input_units, output_units

class ReLU(Layer):
    # ReLU activation unit
    def __init__(self):
        pass

    def forward(self, input):
        return np.maximum(0, input)

    def isWeighted(self):
        return False

    def getDimensions(self):
        return 1, 1


class Sigmoid(Layer):
    # Sigmoid activation unit
    def __init__(self):
        pass

    def forward(self, input):
        return 1.0/(1.0 + np.exp(-input))

    def isWeighted(self):
        return False

    def getDimensions(self):
        return 1, 1

class NeuralNetwork():
    def __init__(self):
        self.layers = list()

    def calculate(self, input):
        for layer in self.layers:
            input = layer.forward(input)

        return input

    def addLayer(self, layer):
        self.layers.append(layer)
