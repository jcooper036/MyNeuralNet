#!/usr/bin/env python3
from abc import abstractmethod
import numpy as np
from MyNeuralNet.layer import Layer

class ActivationLayer(Layer):
    # input_shape = (1,i)   i the numer of of input neurons
    def __init__(self, input_shape, activation, activation_prime):
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input):
        """Returns the activated input"""
        self.input = input
        self.output = self.activation(self.input)
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        """Returns the input_error=dE/dX for a given output error=dE/dY
           learning_rate is unused, but here for consistency"""
        return self.activation_prime(self.input) * output_error
