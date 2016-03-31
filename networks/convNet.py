'''
Dora Jambor
Basic implementation of a convolutional network in Theano
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

import theano
import theano.tensor as T

# Activation functions
def linear(z): return z
def reLU(z): return max(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

#### Constants
GPU = True
if GPU:
    print "Trying to run under a GPU.  If this is not desired, then modify "+\
        "network3.py\nto set the GPU flag to False."
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print "Running with a CPU.  If this is not desired, then the modify "+\
        "network3.py to set\nthe GPU flag to True."





class FullyConnectedLayer(object):

	def __init__(self, n_in, n_out, activationFunc = sigmoid, p_dropout=0.0):
		self.n_in = n_in
		self.n_out = n_out
		self.activationFunc = activationFunc
		self.p_dropout = p_dropout
		# Weights & Biases
		self.weights = theano.shared(
			np.asarray(
				np.random.normal(loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in,n_out)),
				dtype=theano.config.floatX),
			name='weights', borrow=True)
		self.biases = theano.shared(
			np.asarray(
				np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
				dtype=theano.config.floatX),
			name='biases', borrow=True)
		self.param = [self.weights,self.biases]

	def calc_input(self, inPut, input_dropout, mini_batch_size):
		self.inPut = inPut.reshape((mini_batch_size, self.n_out))
		self.output = self.activationFunc((1-self.p_dropout)*T.dot(self.weights, self.inPut) + self.biases)
	    self.y_out = T.argmax(self.output, axis=1)
		self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)


class ConvPoolLayer(object):
	def __init__(self, filter_shape, image_shape, poolsize=(2,2), activationFunc=sigmoid):
		'''
		filter_shape = (num of filters, num of input map, filter height, filter width)
		image_shape = (mini_batch_size, num of input map, im height, im width)
		'''
		self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activationFunc = activationFunc










