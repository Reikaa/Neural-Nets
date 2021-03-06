# Dora Jambor
# MNIST digit recognition 
# following Michael Nielsen's book on Neural Network and Deep Learning

'''
Plots how well my network2 (nesterov) is doing with different values of eta.
'''

# Standard library
import json
import random
import sys

# My library
sys.path.append('../networks/')
import mnist_loader
import nnetwork2

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np

# Constants
LEARNING_RATES = [0.025, 0.25, 2.5]
COLORS = ['#2A6EA6', '#FFCD33', '#FF7033']
NUM_EPOCHS = 30

def run_networks():
    # Make results more easily reproducible    
    random.seed(12345678)
    np.random.seed(12345678)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = nnetwork2.Network([784, 30, 10], 0.5)
    results = []
    for eta in LEARNING_RATES:
        print "\nTrain a network using eta = "+str(eta)
        results.append(net.gradientDescent(training_data, 10, eta, NUM_EPOCHS,
                    test_data=test_data))
    f = open("eta_graph2.json", "w")
    json.dump(results, f)
    f.close()

def plot():
    f = open("eta_graph2.json", "r")
    results = json.load(f)
    results = [results/100 for e in results]
    f.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for eta, result, color in zip(LEARNING_RATES,results, COLORS):
        print len(result), len(COLORS), len(np.arange(NUM_EPOCHS))
        ax.plot(np.arange(NUM_EPOCHS), result, "o-", color=color, label="$\eta$ = "+str(eta))
    ax.set_xlim([0, NUM_EPOCHS])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy in %')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    run_networks()
    plot()