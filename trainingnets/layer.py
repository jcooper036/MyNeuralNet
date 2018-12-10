#!/usr/bin/env python3
from abc import abstractmethod

class Layer:
    """
    Layer's are implimented individually. They 
    can take an input and pass an output. They
    can also take the derivative of the error
    with respect to their output and calculate
    the derivative of the error with respect to
    their input
    """
    
    def __init__(self):
        """Basic constructor function"""
        self.input = None
        self.output = None
        self.input_shape = None
        self.output_shape = None
    
    @abstractmethod
    def forward_propagation(self, input):
        """Propogating the signal through the layer"""
        raise NotImplementedError
    
    @abstractmethod
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
    
    