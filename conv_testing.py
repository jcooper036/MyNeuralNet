#!/usr/bin/env python3
import numpy as np
import trainingnets as mnn
import random

## training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = mnn.Network()
net.add(mnn.FClayer((1,2), (1,3)))
net.add(mnn.ActivationLayer((1,3), mnn.tanh, mnn.tanh_prime))
net.add(mnn.FClayer((1,3), (1,1)))
net.add(mnn.ActivationLayer((1,1), mnn.tanh, mnn.tanh_prime))

# train
net.use(mnn.mse, mnn.mse_prime)
net.fit(x_train, y_train, epochs=100, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)