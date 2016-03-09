# Dora Jambor
# MNIST digit recognition 
# following Michael Nielsen's book on Neural Network and Deep Learning

'''
Plots how well our neural net is doing with different values of eta.
'''

# Standard library
import json
import random
import sys

# My library
sys.path.append('../src/')
import mnist_loader
import nnetwork1

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
    results = []
    for eta in LEARNING_RATES:
        print "\nTrain a network using eta = "+str(eta)
        net = nnetwork1.Network([784, 30, 10], 0.5)
        results.append(
            net.gradientDescent(training_data, 10, eta, NUM_EPOCHS,
                    test_data=test_data))
    f = open("eta_graphs.json", "w")
    json.dump(results, f)
    f.close()

def plot():
    f = open("eta_graphs.json", "r")
    results = json.load(f)
    f.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for eta, result, color in zip(LEARNING_RATES, results, COLORS):
        _, _, training_cost, _ = result
        ax.plot(np.arange(NUM_EPOCHS), training_cost, "o-",
                label="$\eta$ = "+str(eta),
                color=color)
    ax.set_xlim([0, NUM_EPOCHS])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost')
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    run_networks()
    plot()