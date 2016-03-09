# Dora Jambor
# MNIST digit recognition 
# following Michael Nielsen's book on Neural Network and Deep Learning

'''Neural net adjusted by momentum. Accuracy goes up to 95%, but it fluctuates around it through the last 15 epochs. 
15 seconds each epoch. Uncomment the draw function to see the ascii drawing 
of each digit and the corresponding prediction.
Run by unit_test1.py, play with accuracy by adjusting the learning rate and epochs.'''

import numpy as np
import random
import math
import sys
import time


class Network:            
    '''
    Neural Network class
    Steps: 
        - give some input
        - feedforward -> get activation vector

    '''
    def __init__(self, sizes, mu):
        self.layers = len(sizes)
        self.sizes = sizes   
        self.mu = mu                                                           # list of neurons on each layer
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]     # create array of weights with random numbers
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]                         # create array of biases with random numbers
        self.vb = [np.zeros(b.shape) for b in self.biases]
        self.vw = [np.zeros(w.shape) for w in self.weights]

    def feedForward(self, a):
        '''Calculates the activation vector from all inputs from previous layer.
        - redefine input vector if not fully connvected ?
        - or set weights and biases to zero where no connection was made?
        - layers must be at least 1 - to start from first hidden layer'''
        for b, w in zip(self.biases, self.weights):                                      # you loop through each neuron on each layer
            a = sigmoid(np.dot(w, a) + b)                                                # to calculate activation vector in the last layer
        return a                                                                         # z = w . x + b, a is the last output vector

    def gradientDescent(self, trainingSet, batch_size, learningRate, epochs,test_data=None):
        '''
        You have some data from the trainingSet with (x,y) tuples with x
        being the training input and y being the desired output ->classification.
        You can use stochastic gradient descent with smaller batch sizes.
        '''
        if test_data: n_test = len(test_data)
        trainingSize = len(trainingSet)

        # repeat this until finding 'reliable' accuracy between desired and real outcomes
        for i in xrange(epochs):
            print "Starting epoch"
            start = time.time()
            random.shuffle(trainingSet)
            # create smaller samples to do your computations on                                                   
            batches = [trainingSet[k:k + batch_size] for k in xrange(0, trainingSize, batch_size)]                                                                              
            # update each image in each batch
            for batch in batches:
                self.update(batch, learningRate)
            # take the 10K images that were reserved for validation and check accuracy
            print "Validating per epoch..."
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    i, self.validate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)
            timer = time.time() - start
            print "Estimated time: ", timer

    def update(self, batch, learningRate):
        '''
        Backpropagate will return derivates of the cost function w.r.t. b 
        and w.r.t. w for each neuron, which will then be used to calculate 
        the new biases and weights matrices. 
        biases = biases - learningRate * deltaB
        weights = weights -learningRate * deltaW * input
        '''
        # loop through each picture in the given batch: x is input, y is desired output
        for x,y in batch:

            # backpropagate to get (C/b)' and (C/w)' - two vectors
            deltaBiases, deltaWeights = self.backprop(x,y)

            # calculate new biases and weights
            self.vb = [self.mu * vb - learningRate * db/len(batch) for vb,db in zip(self.vb, deltaBiases)]
            self.vw = [self.mu * vw - learningRate * dw/len(batch) for vw,dw in zip(self.vw, deltaWeights)]
            self.weights = [w + vw for w, vw in zip(self.weights, self.vw)]
            self.biases = [b + vb for b, vb in zip(self.biases, self.vb)]


        # update biases and weights matrices

    def backprop(self, x, y):
        ''' Takes (x,y) where x is the pixel from the training image, y is the desired outcome
        and returns a tuple of two vectors of the same shape as biases and weights.
        Steps: 
        1. feedforward while saving each z value in z_vectors
        2. calculate delta on last layer and backpropagate to update delta_b and delta_w
        '''
        delta_b = [np.zeros(b.shape) for b in self.biases]                             # Set up numpy vector to store bias deltas
        delta_w = [np.zeros(w.shape) for w in self.weights]                            # Set up numpy vector to store weight deltas

        a = x                         # x is the pixel input from the training image (at the input layer)
        z_vectors = []
        all_activations = [x]         # store all input vectors

        # First step: FEEDFORWARD
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b      # this is the weighted input
            z_vectors.append(z)       # store all z vectors - last vector is computed right before last layer
            a = sigmoid(z)            # calculate the activation function in the last layer
            all_activations.append(a) # store all act. vectors in this list
        
        # First equation -> calculate delta at final layer from the cost function
        delta = (all_activations[-1] - y) * sigmoid_prime(z_vectors[-1])
        delta_b[-1] = delta
        delta_w[-1] = np.dot(delta, all_activations[-2].transpose())

        # Second step: OUTPUT ERROR
        for l in range(2, self.layers):
            z = z_vectors[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp                  # backprop to calculate error (delta) at layer - 1
            delta_b[-l] = delta
            delta_w[-l] = np.dot(delta, all_activations[-l-1].transpose())
        return delta_b, delta_w

    def validate(self, test_data):
        ''' Go through the data you set aside for validation, 
        take all outcomes (x vector) for each picture and get the INDEX of the highest 
        outcome -> the outcome that fired the most. 
        Then check how many images you'll get the correct result for.
        '''
        test_results = [(np.argmax(self.feedForward(x)),y) for x, y in test_data]
        # draw(test_data, test_result)                                                    # draw images in command line
        return sum(int(x == y) for x, y in test_results)                                # check for accuracy

def draw(test_data, test_result):
        # print len(test_data[0])       # 2 tuple inputs
        # print len(test_data[0][0])    # 784x1 pixels => should reshape to 28 * 28
        # print test_data[0][0][0][0]   # pixel value
        i = 0
        for j in range(len(test_data)):
            for num in np.nditer(test_data[j][0]):
                if i%28 == 0:
                    print "\n",
                i += 1
                if num < 0.1:
                    sys.stdout.write(' ')
                else:
                    sys.stdout.write('x')
            print "\nMy output:", test_results[j][0]
            print "The desired class is:", test_data[j][1]

def sigmoid(z):
        '''
        Arbitrary choice of function. Use sigmoid or relu in this case.
        This will help us to create the activation vector.
        You need something that can be conveniently backpropagated.
        '''
        # return math.log(1+math.exp(z))
        return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    ''' Returns the derivative of sigmoid(z = w.x + b) w.r.t. z'''
    return sigmoid(z)*(1-sigmoid(z))





