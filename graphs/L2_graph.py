# Dora Jambor
# MNIST digit recognition 
# following Michael Nielsen's book on Neural Network and Deep Learning

# Reminder: try to make the other graph into an additional file.

'''
Plots network3 with L2 regularization.
'''

# Standard library
import json
import random
import sys

# My library
sys.path.append('../networks/')
import mnist_loader
import nnetwork3

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np

# Constants
eta = 2.5
COLORS = ['#2A6EA6', '#FFCD33','#FF7033']
NUM_EPOCHS = 30
INPUT_NEURONS = 784
HIDDEN_NEURONS = 30
OUTPUT_NEURONS = 10
BATCH_SIZE = 10
lmbda = [0.1, 1.5, 3.0]

def run_networks():
    # Make results more easily reproducible    
    random.seed(12345678)
    np.random.seed(12345678)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    # instantiate network
    net = nnetwork3.Network([INPUT_NEURONS,HIDDEN_NEURONS,OUTPUT_NEURONS])
    # run SGD
    results = []
    for lmd in lmbda:
        results.append(net.gradientDescent(training_data, BATCH_SIZE, eta, NUM_EPOCHS, lmd,
                    test_data=test_data))
    f = open("L2_graph.json", "w")
    json.dump(results, f)
    f.close()

def plot():
    f = open("L2_graph.json", "r")
    results = json.load(f)
    # results = [e/100.0 for e in results]
    # print results
    f.close()
    fig = plt.figure()
    plt.title("Regularized network")
    ax = fig.add_subplot(111)
    for lmd, result, color in zip(lmbda,results, COLORS):
        ax.plot(np.arange(NUM_EPOCHS), result, "o-", color=color, label=u"\u03BB = "+str(lmd))

    ax.set_xlim([0, NUM_EPOCHS])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy in %')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    # run_networks()
    plot()