#!/usr/bin/env python3
import numpy as np
import MyNeuralNet as mnn
import random


## activation and activation prime functions
def tanh(x):
    """Activaiton function"""
    return np.tanh(x)
def tanh_prime(x):
    """Activation prime function"""
    return 1-np.tanh(x)**2

## loss function an derivative
def mse(y_true, y_pred):
    """Loss function (Mean Squared Error)"""
    return np.mean(np.power(y_true - y_pred, 2))
def mse_prime(y_true, y_pred):
    """Derivative of loss function"""
    return 2*(y_pred - y_true) / y_true.size

## training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = mnn.Network()
net.add(mnn.FClayer((1,2), (1,3)))
net.add(mnn.ActivationLayer((1,3), tanh, tanh_prime))
net.add(mnn.FClayer((1,3), (1,1)))
net.add(mnn.ActivationLayer((1,1), tanh, tanh_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=100, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)


## can it figure out how to add 0.5 to each number?
x_train = []
y_train = []

training_samples = 10

for i in range(training_samples):
    x_train.append(np.random.rand(1,2)-0.5)
    y_train.append(x_train[i] + 0.5)

# network
net = mnn.Network()
net.add(mnn.FClayer((1,2), (1,3)))
net.add(mnn.ActivationLayer((1,3), tanh, tanh_prime))
net.add(mnn.FClayer((1,3), (1,2)))
net.add(mnn.ActivationLayer((1,2), tanh, tanh_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=100, learning_rate=0.1)

# test
final_test = np.random.rand(1,2)-0.5
out = net.predict(final_test)
print(final_test)
print(out)


## what about if it has to add 0.5 or subtract 0.5, depending on if the 3rd position is 1 or -1 ?
x_train = []
y_train = []

training_samples = 10

for i in range(training_samples):
    plusORminus = random.randint(0,1)
    x_train.append(np.array([[random.random() - 0.5, random.random() - 0.5, plusORminus]] ))
    if x_train[i][0][2] == 1:
        y_train.append(np.array( [[x_train[i][0][0] + 0.5, x_train[i][0][1] + 0.5]] ))
    if x_train[i][0][2] == 0:
        y_train.append(np.array( [[x_train[i][0][0] - 0.5, x_train[i][0][1] - 0.5]] ))

# network
net = mnn.Network()
net.add(mnn.FClayer((1,3), (1,4)))
net.add(mnn.ActivationLayer((1,4), tanh, tanh_prime))
net.add(mnn.FClayer((1,4), (1,2)))
net.add(mnn.ActivationLayer((1,2), tanh, tanh_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
final_test = np.array(np.array([[random.random() - 0.5, random.random() - 0.5, 1]] ))
out = net.predict(final_test)
print(final_test)
print(out)

final_test = np.array(np.array([[random.random() - 0.5, random.random() - 0.5, 0]] ))
out = net.predict(final_test)
print(final_test)
print(out)

