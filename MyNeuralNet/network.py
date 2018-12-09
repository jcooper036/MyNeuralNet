#!/usr/bin/env python3
from MyNeuralNet.layer import Layer

class Network:
    """Container + some processing class for a neural net"""
    
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        """Adds a layer to the network"""
        self.layers.append(layer)
    
    def use(self, loss, loss_prime):
        """Set the loss functions to use"""
        self.loss = loss
        self.loss_prime = loss_prime
    
    def network_pass(self, sample):
        """Runs the forward propagation loop"""
        output = sample
        for layer in self.layers:
            output = layer.forward_propagation(output)
        return output
    
    def predict(self, input):
        """Predict and output given an input"""
        # dimensions first
        samples = len(input)
        result = []

        # run network over all samples
        for i in range (samples):
            # forward propagation
            result.append(self.network_pass(input[i]))
        
        return result
    
    def fit(self, x_train, y_train, epochs, learning_rate):
        """Train the network using x_train inputs and y_train answers"""
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # foward propagation
                output = self.network_pass(x_train[j])

                # comput the loss function
                err += self.loss(y_train[j], output)

                # back propagation
                error = self.loss_prime(y_train[j], output)

                # loop from the end of the network to the begining
                for layer in reversed(self.layers):
                    # backpropagate dE
                    error = layer.backward_propagation(error, learning_rate)

            # calculate avergae error on all samples
            err /= samples
            print('epoch %d%d   error=%f' % (i+1, epochs, err))