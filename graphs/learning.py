# Dora Jambor
# MNIST digit recognition 
# following Michael Nielsen's book on Neural Network and Deep Learning

'''
Plots the first network using hyperparameters suggested in the book.
'''

# Standard library
import json
import random
import sys

# My library
sys.path.append('../networks/')
import mnist_loader
import nnetwork

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np

# Constants
eta = 2.5
color = '#2A6EA6'
NUM_EPOCHS = 30
INPUT_NEURONS = 784
HIDDEN_NEURONS = 30
OUTPUT_NEURONS = 10
BATCH_SIZE = 10

def run_networks():
    # Make results more easily reproducible    
    random.seed(12345678)
    np.random.seed(12345678)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    # instantiate network
    net = nnetwork.Network([INPUT_NEURONS,HIDDEN_NEURONS,OUTPUT_NEURONS])
    # run SGD
    results = net.gradientDescent(training_data, BATCH_SIZE, eta, NUM_EPOCHS,
                    test_data=test_data)
    f = open("learning.json", "w")
    json.dump(results, f)
    f.close()

def plot():
    f = open("learning.json", "r")
    results = json.load(f)
    results = [e/100.0 for e in results]
    print results
    f.close()
    fig = plt.figure()
    plt.title("Network I.")
    ax = fig.add_subplot(111)
    ax.plot(np.arange(NUM_EPOCHS), results, "o-", color=color)
    ax.set_xlim([0, NUM_EPOCHS])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy in %')

    plt.show()

if __name__ == "__main__":
    # run_networks()
    plot()