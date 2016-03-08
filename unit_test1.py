# Dora Jambor
# MNIST digit recognition 
# following Michael Nielsen's book on Neural Network and Deep Learning

'''Test for nneuwork1.py for MNIST dataset. This NN is adjusted by momentum'''

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import nnetwork1
# input layers are for the MNIST dataset where each image is of 28 x 28
net = nnetwork1.Network([784, 30, 10], 0.9)
# the arguments are the following: training data, batch size, learning rate and number of epochs
net.gradientDescent(training_data, 32, 3.0/2, 30, test_data=test_data)


