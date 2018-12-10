#!/usr/bin/env python3
import numpy as np

def mse(y_true, y_pred):
    """Loss function (Mean Squared Error)"""
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    """Derivative of loss function Mean Squared Error"""
    return 2*(y_pred - y_true) / y_true.size
