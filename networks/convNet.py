'''
Dora Jambor
Basic implementation of a convolutional network in Theano
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

import theano
import theano.tensor as T
import gzip

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

def load_into_shared(filename="../networks/mnist.pkl.gz"):
	''' Loads data into Theano shared variables so that theano can copy data to the GPU if available '''
	f = gzip.open(filename, 'rb')
	training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    def shared(data):
    	shared_x = theano.shared(
    		np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
    	shared_y = theano.shared(
    		np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
    	return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]

class Network(object):

	def __init__(self, layers, mini_batch_size):
		self.layers = layers
		self.mini_batch_size = mini_batch_size
		self.params = [param for layer in self.layers for param in layer.params]
		# T.matrix is a 2 dimensional tensor of float32's -> INPUT PIXEL FLOATS
		self.x = T.matrix("x")
		# T.ivector is a one dimensional int32 vector in theano -> LABELS
		self.y = T.ivector("y")
		input_layer = self.layers[0]
		# where is set_inpt defined?
		input_layer.set_inpt(self.x, self.x, self.mini_batch_size)
		# set inputs and outputs on each layer!!!
		for i in xrange(1, len(self.layers)):
			prev_layer, current_layer = self.layers[i-1], self.layers[i]
			layer.set_inpt(prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
		# set outputs on last layer
		self.output = self.layers[-1].output
		self.output_dropout = self.layers[-1].output_dropout

	def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
		'''
		For epoch:
			- shuffle
			- batches
			For batch:
				- update
			- validate

		'''
		training_x, training_y = training_data
		validation_x, validation_y = validation_data
		test_x, test_y = test_data
		# why not self.mini_batch_size ??
		# what's size() and why not len?
		batchNum_training = len(training_data) / mini_batch_size
		batchNum_validation = len(validation_data) / mini_batch_size
		batchNum_test = len(test_data) / mini_batch_size

		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Update & Cost parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# get the sum of squared weights in each layer
		L2_sq_norm = sum([(layer.weights**2).sum() for layer in self.layers])
		# C = C(woL2) + lambda*sum(weights) / num of batch iterations
		cost = self.layers[-1].cost(self) + 0.5*lmbda * L2_sq_norm/batchNum_validation
		grads = T.grad(cost, self.params)
		updates = [(param, param-eta*grad) for param,grad in zip(self.params, grads)]

		i = T.iscalar() # index for mini-bathch

		# set up of training functions (no func calls yet)
		training









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

        # num of filters x receptive field (5 x 5) / poolsize 
        # --> num of neurons after the pool layer
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))

        # weights & biases
        # weights are squashed by variance of 1/n_out
        self.weights = theano.shared(np.asarray(
        	np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
        	dtype=theano.config.floatX), borrow = True)
        self.biases = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.weights, self.biases]

    def set_inpt(self,inpt,inpt_dropout, mini_batch_size):
    	# create an input matrix shaped by: batch size x 1 x 28 x 28
    	# this input layer will account for all images in the batch
    	self.inpt = inpt.reshape(self.image_shape)
    	#
    	conv_out = conv.conv2d(
    		input=self.inpt, filters=self.weights, filter_shape=self.filter_shape,
    		image_shape=self.image_shape)
    	pooled_out = downsample.max_pool_2d(
    		input=conv_out, ds=self.poolsize, ignore_border=True)
    	self.output = self.activationFunc(
    		pooled_out + self.biases.dimshuffle('x',0, 'x', 'x'))
    	# no dropout on convolutional layer
    	self.output_dropout = self.output


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

	def calc_input(self, inPut, inpt_dropout, mini_batch_size):
		'''
		n_in is the number of input neurons (pixels)
		Weights are a matrix of (n_in, n_out)
		So dot.product of input x weights -> (batch_size, n_in).(n_in x n_out)
		self.input is a matrix of 
		'''
		self.inPut = inPut.reshape((mini_batch_size, self.n_in))
		self.output = self.activationFunc((1-self.p_dropout)*T.dot(self.weights, self.inPut) + self.biases)
	    # take neuron with max output
	    self.y_out = T.argmax(self.output, axis=1)
	    # update inputs with dropout
		self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
		# calculate output with dropout -> using the dropout inputs
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.weights) + self.biases)

    def accuracy(self, y):
        ''' returns accuracy of mini batch '''
    	# T.eq returns a variable representing the result of logical equality (a==b).
     	return T.mean(T.eq(y,self.y_out))

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.weights = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='weights', borrow=True)
        self.biases = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='biases', borrow=True)
        self.params = [self.weights, self.biases]

    def set_inpt(self, inPut, inpt_dropout, mini_batch_size):
        self.inPut = inPut.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inPut, self.weights) + self.biases)
        # return highest output
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)  
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.weights) + self.biases)

    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    ''' size of layer is the nmber of neurons on the layers from which you will draw
    k elements with a probability of success (1-P_dropout). 
    Mask basically returns the variables that are the successes.
    Check what layer does -> this should return the layer without the dropped out neurons.
    '''
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    # binomial returns the neurons that are not dropped out!!!!!
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    # cast returns mask with type dfined as floatX (float32)
    return layer*T.cast(mask, theano.config.floatX)









