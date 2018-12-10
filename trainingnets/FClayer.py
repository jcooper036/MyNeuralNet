#!/usr/bin/env python3
from abc import abstractmethod
import numpy as np
from trainingnets.layer import Layer

class FClayer(Layer):
    """Initialize a fully connected layer"""
    # input_shape = (1,i)  i is the number of input neurons
    # output_shape = (1,j) j is the number of output neurons
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = np.random.rand(input_shape[1], output_shape[1]) - 0.5
        self.bias = np.random.rand(1, output_shape[1]) - 0.5

    def forward_propagation(self, input):
        """returns output for a given input"""
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        """compute dE/dW, dE/dB for given output_error = dE/dY. Returns input_error = dE/dX"""
        input_error = np.dot(output_error, self.weights.T)
        dWeights = np.dot(self.input.T, output_error)
        dBias = output_error

        # update parameters
        self.weights -= learning_rate * dWeights
        self.bias -= learning_rate * dBias
        
        return input_error