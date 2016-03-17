# Dora Jambor
# MNIST digit recognition 
# following Michael Nielsen's book on Neural Network and Deep Learning

'''
Plots network1 (momentum) to see how it is doing with different values of eta given mu=0.5.
'''

# Standard library
import json
import random
import sys

# My library
sys.path.append('../networks/')
import mnist_loader
import nnetwork1

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np

# Constants
LEARNING_RATES = [0.025, 0.25, 2.5]
COLORS = ['#2A6EA6', '#FFCD33', '#FF7033']
NUM_EPOCHS = 30
INPUT_NEURONS = 784
HIDDEN_NEURONS = 30
OUTPUT_NEURONS = 10
batch_size = 10
mu = 0.5

def run_networks():
    # Make results more easily reproducible    
    random.seed(12345678)
    np.random.seed(12345678)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = nnetwork1.Network([INPUT_NEURONS,HIDDEN_NEURONS,OUTPUT_NEURONS], mu)
    results = []
    for eta in LEARNING_RATES:
        print "\nTrain a network using eta = "+str(eta)
        results.append(net.gradientDescent(training_data, batch_size, eta, NUM_EPOCHS,
                    test_data=test_data))
    f = open("eta_graph1.json", "w")
    json.dump(results, f)
    f.close()

def plot():
    f = open("eta_graph1.json", "r")
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