# Dora Jambor
# MNIST digit recognition 
# following Michael Nielsen's book on Neural Network and Deep Learning

'''Test for nn3_weight.py for MNIST dataset. This NN is adjusted by L2 regularization and better weight initialization.'''

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import nn3_weight

eta = 1.5
NUM_EPOCHS = 30
INPUT_NEURONS = 784
HIDDEN_NEURONS = 30
OUTPUT_NEURONS = 10
BATCH_SIZE = 10
lmbda = 0.1

# input layers are for the MNIST dataset where each image is of 28 x 28
net = nn3_weight.Network([INPUT_NEURONS,HIDDEN_NEURONS,OUTPUT_NEURONS])
# the arguments are the following: training data, batch size, learning rate and number of epochs
net.gradientDescent(training_data, BATCH_SIZE, eta, NUM_EPOCHS, lmbda,
                    test_data=test_data)


