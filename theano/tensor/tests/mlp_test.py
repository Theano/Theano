"""
This is a minimized version of the mlp.py in the tutorial. We removed stuff that make this mlp don't work.
But this test a bug that we saw. This bug made the Shape_i object not being lifted, that caused the CrossentropySoftmax... op not being inserted.
"""
from __future__ import absolute_import, print_function, division

__docformat__ = 'restructedtext en'


from collections import OrderedDict
import numpy as np

import theano
import theano.tensor as T


def gen_data():

    # generate the dataset
    train_set = (np.asarray(np.random.rand(10000, 784), dtype='float32'),
               np.asarray(np.random.rand(10000)*10, dtype='int64'))
    valid_set = (np.asarray(np.random.rand(10000, 784), dtype='float32'),
               np.asarray(np.random.rand(10000)*10, dtype='int64'))
    test_set = (np.asarray(np.random.rand(10000, 784), dtype='float32'),
               np.asarray(np.random.rand(10000)*10, dtype='int64'))
    def shared_dataset(data_xy):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch every time
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x,  test_set_y  = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, name_prefix=''):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
                                name=name_prefix+'W')

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W))

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e., number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain [0,1,2,... n-1]
        # T.log(self.p_y_given_x) is a matrix of Log-Probabilities (call it LP) with one row per example and one column per class
        # LP[T.arange(y.shape[0]),y] is a vector v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ..., LP[n-1,y[n-1]]]
        # and T.mean(LP[T.arange(y.shape[0]),y]) is the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return T.log(self.p_y_given_x[T.arange(y.shape[0]), y])


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation=T.tanh, name_prefix=''):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                              layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from -6./sqrt(n_in+n_hidden) and 6./sqrt(n_in+n_hidden)
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        W_values = np.asarray( rng.uniform( \
              low=-np.sqrt(6./(n_in+n_out)), \
              high=np.sqrt(6./(n_in+n_out)), \
              size=(n_in, n_out)), dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values, name=name_prefix+'W')

        self.output = T.dot(input, self.W)
        # parameters of the model
        self.params = [self.W]


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermidiate layers usually have as activation function thanh or the
    sigmoid function (defined here by a ``SigmoidalLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will
        # translate into a TanhLayer connected to the LogisticRegression
        # layer; this can be replaced by a SigmoidalLayer, or a layer
        # implementing any other nonlinearity
        self.hiddenLayer = HiddenLayer(rng=rng, input=input,
                                 n_in=n_in, n_out=n_hidden,
                                 activation=T.tanh, name_prefix='hid_')

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
                                    input=self.hiddenLayer.output,
                                    n_in=n_hidden,
                                    n_out=n_out, name_prefix='log_')

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params


def test_mlp():
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                         http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """
    datasets = gen_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x , test_set_y  = datasets[2]

    batch_size = 100    # size of the minibatch

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches  = test_set_x.get_value(borrow=True).shape[0]  / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    # print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x     = T.matrix('x')  # the data is presented as rasterized images
    y     = T.ivector('y')  # the labels are presented as 1D vector of
                           # [int] labels

    rng = np.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP( rng=rng, input=x, n_in=28*28, n_hidden=500, n_out=10)

    # the cost we minimize during training is the negative log likelihood of
    # the model.
    # We take the mean of the cost over each minibatch.
    cost = classifier.negative_log_likelihood(y).mean()

    # compute the gradient of cost with respect to theta (stored in params)
    # the resulting gradients will be stored in a list gparams
    gparams = []
    for param in classifier.params:
        gparam  = T.grad(cost, param)
        gparams.append(gparam)

    # Some optimizations needed are tagged with 'fast_run'
    # TODO: refine that and include only those
    mode = theano.compile.get_default_mode().including('fast_run')

    updates2 = OrderedDict()

    updates2[classifier.hiddenLayer.params[0]] = T.grad(cost, classifier.hiddenLayer.params[0])
    train_model = theano.function( inputs=[index],
            updates=updates2,
            givens={
                x: train_set_x[index*batch_size:(index+1)*batch_size],
                y: train_set_y[index*batch_size:(index+1)*batch_size]},
            mode=mode)
    # print 'MODEL 1'
    #theano.printing.debugprint(train_model, print_type=True)
    assert any([isinstance(i.op, T.nnet.CrossentropySoftmax1HotWithBiasDx) for i in train_model.maker.fgraph.toposort()])

    # Even without FeatureShape
    train_model = theano.function( inputs=[index],
            updates=updates2,
            mode=mode.excluding('ShapeOpt'),
            givens={
                x: train_set_x[index*batch_size:(index+1)*batch_size],
                y: train_set_y[index*batch_size:(index+1)*batch_size]})
    # print
    # print 'MODEL 2'
    #theano.printing.debugprint(train_model, print_type=True)
    assert any([isinstance(i.op, T.nnet.CrossentropySoftmax1HotWithBiasDx) for i in train_model.maker.fgraph.toposort()])

if __name__ == '__main__':
    test_mlp()
