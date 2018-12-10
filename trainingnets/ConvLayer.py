#!/usr/bin/env python3
import numpy as np
from scipy import signal
from trainingnets.layer import Layer

class ConvLayer(Layer):
    """
    Convolutional layer, use when the structure between two layers is known
    input shape = (i, j, d)
    kernal shape = (m, n)
    layer_depth = output_depth
    """
    def __init__(self, input_shape, kernal_shape, layer_depth):
        self.input_shape = input_shape
        self.input_depth = input_shape[2]
        self.kernal_shape = kernal_shape
        self.layer_depth = layer_depth
        self.output_shape = (input_shape[0]-kernel_shape[0]+1, input_shape[1]-kernel_shape[1]+1, layer_depth)
        self.weights = np.random.rand(kernel_shape[0], kernel_shape[1], self.input_depth, layer_depth) - 0.5
        self.bias = np.random.rand(layer_depth) - 0.5
        print(self.weights, '\n')


    def forward_propagation(self, input):
        """returns output for a given input"""
        self.input = input
        self.output = np.zeros(self.output_shape)

        for k in range(self.layer_depth):
            for d in range(self.input_depth):
                self.output[:,:,k] += signal.correlate2d(self.input[:,:,d], self.weights[:,:,d,k], 'valid') + self.bias[k]

        print(self.input, '\n')
        print(self.output, '\n')
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        """computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX."""
        in_error = np.zeros(self.input_shape)
        dWeights = np.zeros((self.kernel_shape[0], self.kernel_shape[1], self.input_depth, self.layer_depth))
        dBias = np.zeros(self.layer_depth)

        for k in range(self.layer_depth):
            for d in range(self.input_depth):
                in_error[:,:,d] += signal.convolve2d(output_error[:,:,k], self.weights[:,:,d,k], 'full')
                dWeights[:,:,d,k] = signal.correlate2d(self.input[:,:,d], output_error[:,:,k], 'valid')
            dBias[k] = self.layer_depth * np.sum(output_error[:,:,k])

        self.weights -= learning_rate*dWeights
        self.bias -= learning_rate*dBias
        
        return in_error