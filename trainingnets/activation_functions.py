#!/usr/bin/env python3
import numpy as np

## activation and activation prime functions
def tanh(x):
    """Activaiton function"""
    return np.tanh(x)
def tanh_prime(x):
    """Activation prime function"""
    return 1-np.tanh(x)**2

def sigmoid(x):
    return 1. / (1. + np.exp(-x))
def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))