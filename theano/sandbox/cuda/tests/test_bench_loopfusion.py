"""
This file is based on hpu.nns.driver_kouh of Oct 22 2009.

It is meant to be used to benchmark loop fusion optimizations.

"""
# this experiments are designed to use file-based configuration
# rather than db-based configuration.
# so state is ignored

# since this job is not restartable, channel is also ignored
import logging, StringIO, time, sys

import numpy

import theano
from theano.compile import shared, pfunc
from theano import tensor
from theano.tensor.nnet import softplus
from theano.sandbox.softsign import softsign

_logger = logging.getLogger('theano.sandbox.cuda.tests.test_bench_loopfusion')

def _shared_uniform(rng, low, high, size, dtype, name=None):
    return shared(
            theano._asarray(
                rng.uniform(low=low, high=high, size=size),
                dtype=dtype), name)

class Kouh2008(object):
    """WRITEME

    :param x: a list of N non-negative tensors of shape (n_examples, n_out)
    :param w: a list of N output weights of shape (n_out, )
    :param p: a tensor of exponents of shape (n_out,)
    :param q: a tensor of exponents of shape (n_out,)
    :param r: a tensor of exponents of shape (n_out,)
    :param k: a tensor of biases of shape (n_out,)

    output - a tensor of activations of shape (n_examples, n_out)
    """

    def __init__(self, w_list, x_list, p, q, r, k, params, updates, eps=1.0e-6):
        """Transcription of equation 2.1 from paper (page 1434).
        """
        if len(w_list) != len(x_list):
            raise ValueError('w_list must have same len as x_list')
        output = (sum(w * tensor.pow(x, p) for (w,x) in zip(w_list, x_list)))\
                / (theano._asarray(eps, dtype=k.type.dtype) + k + tensor.pow(sum(tensor.pow(x, q) for x in x_list), r))

        assert output.type.ndim == 2
        self.__dict__.update(locals())
        del self.__dict__['self']
        _logger.debug('output dtype %s' % output.dtype)

    @classmethod
    def new_expbounds(cls, rng, x_list, n_out, dtype=None, params=[], updates=[], exponent_range=(1.0, 3.0)):
        """
        """
        if dtype is None:
            dtype = x_list[0].dtype
        n_terms = len(x_list)

        def shared_uniform(low, high, size, name):
            return _shared_uniform(rng, low, high, size, dtype, name)

        use_softmax_w = True

        if use_softmax_w:
            w = shared_uniform(low=-.1, high=.1, size=(n_out, n_terms), name='Kouh2008::w')
            w_sm = theano.tensor.nnet.softmax(w)
            w_list = [w_sm[:,i] for i in xrange(n_terms)]
            w_l1 = abs(w).sum()
            w_l2_sqr = (w**2).sum()
        else:
            w_list = [shared_uniform(low=-2.0/n_terms, high=2.0/n_terms, size=(n_out,), name='w_%i'%i)
                    for i in xrange(n_terms)]
            w_l1 = sum(abs(wi).sum() for wi in w_list)
            w_l2_sqr = sum((wi**2).sum() for wi in w_list)

        e_range_low, e_range_high = exponent_range
        e_range_low = theano._asarray(e_range_low, dtype=dtype)
        e_range_high = theano._asarray(e_range_high, dtype=dtype)
        e_range_mag = e_range_high - e_range_low
        if e_range_mag < 0:
            raise ValueError('exponent range must have low <= high')

        p_unbounded = shared_uniform(low=-0.1, high=0.1, size=(n_out,), name='p')
        q_unbounded = shared_uniform(low=-0.1, high=0.1, size=(n_out,), name='q')
        r_unbounded = shared_uniform(low=-0.1, high=0.1, size=(n_out,), name='r')
        k_unbounded = shared_uniform(low=-0.2, high=0.2, size=(n_out,), name='k') # biases

        p = tensor.nnet.sigmoid(p_unbounded) * e_range_mag + e_range_low
        q = tensor.nnet.sigmoid(q_unbounded) * e_range_mag + e_range_low
        r = tensor.nnet.sigmoid(r_unbounded) * \
                theano._asarray(1.0/e_range_low - 1.0/e_range_high, dtype=dtype) \
                + theano._asarray(1.0/e_range_high, dtype=dtype)

        k = softsign(k_unbounded)

        if use_softmax_w:
            rval = cls(w_list, x_list, p, q, r, k,
                    params = [p_unbounded, q_unbounded, r_unbounded, k_unbounded, w] + params,
                    updates=updates)
        else:
            rval = cls(w_list, x_list, p, q, r, k,
                    params = [p_unbounded, q_unbounded, r_unbounded, k_unbounded] + w_list + params,
                    updates=updates)
        rval.p_unbounded = p_unbounded
        rval.q_unbounded = q_unbounded
        rval.r_unbounded = r_unbounded
        rval.k_unbounded = k_unbounded
        rval.exp_l1 = abs(p_unbounded).sum() + abs(q_unbounded).sum() + abs(r_unbounded).sum()
        rval.exp_l2_sqr = (p_unbounded**2).sum() + (q_unbounded**2).sum() + (r_unbounded**2).sum()
        rval.w_l1 = w_l1
        rval.w_l2_sqr = w_l2_sqr
        return rval

    @classmethod
    def new_filters_expbounds(cls, rng, input, n_in, n_out, n_terms, dtype=None, eps=1e-1,
            exponent_range=(1.0, 3.0), filter_range=1.0):
        """Return a KouhLayer instance with random parameters

        The parameters are drawn on a range [typically] suitable for fine-tuning by gradient
        descent.


        :param input: a tensor of shape (n_examples, n_in)

        :type n_in: positive int
        :param n_in: number of input dimensions

        :type n_out: positive int
        :param n_out: number of dimensions in rval.output

        :param nterms: each (of n_out) complex-cell firing rate will be determined from this
        many 'simple cell' responses.

        :param eps: this amount is added to the softplus of filter responses as a baseline
        firing rate (that prevents a subsequent error from ``pow(0, p)``)

        :returns: KouhLayer instance with freshly-allocated random weights.

        """
        if input.type.ndim != 2:
            raise TypeError('matrix expected for input')

        if dtype is None:
            dtype = input.dtype
        _logger.debug('dtype %s' % dtype)

        def shared_uniform(low, high, size, name):
            return _shared_uniform(rng, low, high, size, dtype, name)

        f_list = [shared_uniform(low=-2.0/numpy.sqrt(n_in), high=2.0/numpy.sqrt(n_in), size=(n_in, n_out), name='f_%i'%i)
                for i in xrange(n_terms)]

        b_list = [shared_uniform(low=0, high=.01, size=(n_out,), name='b_%i'%i)
                for i in xrange(n_terms)]
        #x_list = [theano._asarray(eps, dtype=dtype)+softplus(tensor.dot(input, f_list[i])) for i in xrange(n_terms)]
        filter_range = theano._asarray(filter_range, dtype=dtype)
        half_filter_range = theano._asarray(filter_range/2, dtype=dtype)
        x_list = [theano._asarray(filter_range + eps, dtype=dtype)+half_filter_range *softsign(tensor.dot(input, f_list[i]) +
            b_list[i]) for i in xrange(n_terms)]

        rval = cls.new_expbounds(rng, x_list, n_out, dtype=dtype, params=f_list + b_list,
                exponent_range=exponent_range)
        rval.f_list = f_list
        rval.input = input #add the input to the returned object
        rval.filter_l1 = sum(abs(fi).sum() for fi in f_list)
        rval.filter_l2_sqr = sum((fi**2).sum() for fi in f_list)
        return rval

    def img_from_weights(self, rows=None, cols=None, row_gap=1, col_gap=1, eps=1e-4):
        """ Return an image that visualizes all the weights in the layer.
        """

        n_in, n_out = self.f_list[0].value.shape

        if rows is None and cols is None:
            rows = int(numpy.sqrt(n_out))
        if cols is None:
            cols = n_out // rows
            if n_out % rows: cols+=1
        if rows is None:
            rows = n_out // cols
            if n_out % cols: rows+=1

        filter_shape = self.filter_shape
        height = rows * (row_gap + filter_shape[0]) - row_gap
        width = cols * (col_gap + filter_shape[1]) - col_gap

        out_array = numpy.zeros((height, width, 3), dtype='uint8')

        w = self.w.value
        w_col = 0
        def pixel_range(x):
            return 255 * (x - x.min()) / (x.max() - x.min() + eps)

        for r in xrange(rows):
            out_r_low = r*(row_gap + filter_shape[0])
            out_r_high = out_r_low + filter_shape[0]
            for c in xrange(cols):
                out_c_low = c*(col_gap + filter_shape[1])
                out_c_high = out_c_low + filter_shape[1]
                out_tile = out_array[out_r_low:out_r_high, out_c_low:out_c_high,:]

                if c % 3 == 0: # linear filter
                    if w_col < w.shape[1]:
                        out_tile[...] = pixel_range(w[:,w_col]).reshape(filter_shape+(1,))
                        w_col += 1
                if c % 3 == 1: # E filters
                    if w_col < w.shape[1]:
                        #filters after the 3rd do not get rendered, but are skipped over.
                        #  there are only 3 colour channels.
                        for i in xrange(min(self.n_E_quadratic,3)):
                            out_tile[:,:,i] = pixel_range(w[:,w_col+i]).reshape(filter_shape)
                        w_col += self.n_E_quadratic
                if c % 3 == 2: # S filters
                    if w_col < w.shape[1]:
                        #filters after the 3rd do not get rendered, but are skipped over.
                        #  there are only 3 colour channels.
                        for i in xrange(min(self.n_S_quadratic,3)):
                            out_tile[:,:,2-i] = pixel_range(w[:,w_col+i]).reshape(filter_shape)
                        w_col += self.n_S_quadratic
        return Image.fromarray(out_array, 'RGB')

class Config(object):
    use_gpu = True
    dtype='float32'
    dtype2=dtype
    if dtype2=='floatX':
        import theano.config as c
        dtype2 = c.config.get('scalar.floatX')

    rng_seed = 23498

    n_hid = 300
    n_terms = 4

    ft_lr_t0 = 3e-3
    ft_t_decay = 0 # 50 * 5000 # (units of minibatches) by this N'th pass through the training set
    ft_lr_t_decay = 1e-3    # we will have this learning rate
    ft_cost_classif_l1 = 0
    ft_cost_classif_l2 = 0
    ft_cost_in_l1_filter = 0
    ft_cost_in_l2_filter = 0
    ft_cost_in_l1_exp = 0
    ft_cost_in_l2_exp = 0
    ft_cost_in_l1_w = 0
    ft_cost_in_l2_w = 0
    ft_limit_iters = -1
    ft_limit_walltime = 0 # in seconds 60*60*1 #1 hour

    ft_batchsize = 30
    ft_epoch_len = 50000
    ft_status_interval = 50 #property( lambda s:s.ft_epoch_len/s.ft_batchsize)
    ft_validation_interval = property( lambda s:s.ft_epoch_len/s.ft_batchsize)
    ft_ntrain_limit = 0
    ft_test_lag1 = True

    lr = 0.001

if 0:
    # commenting out because this is not really a unit test
    # and it doesn't run correctly because of a deprecated call to cuda.use()
    def test_bench_elemwise(n_iter=1000, **kwargs):
        conf = Config()
        for k in kwargs:
            setattr(conf, k, kwargs[k])

        if conf.use_gpu:
            # Skip test if cuda_ndarray is not available.
            from nose.plugins.skip import SkipTest
            import theano.sandbox.cuda as cuda_ndarray
            if not cuda_ndarray.cuda_available:
                raise SkipTest('Optional package cuda disabled')
            import theano.sandbox.cuda
            theano.sandbox.cuda.use()

        debug=False
        if isinstance(theano.compile.mode.get_default_mode(),
                theano.compile.debugmode.DebugMode):
            debug=True

        # get symbolic train set
        s_lr = theano.tensor.fscalar()
        if not debug:
            sshape = (None, 784)
        else: sshape = (None, 3)
        x = theano.tensor.TensorType(dtype=conf.dtype, broadcastable=(0,0), shape=sshape)()
        y = theano.tensor.lvector()

        rng = numpy.random.RandomState(conf.rng_seed)

        if not debug:
            layer = Kouh2008.new_filters_expbounds(rng, x, x.type.shape[1], conf.n_hid, conf.n_terms)
        else:
            layer = Kouh2008.new_filters_expbounds(rng, x, x.type.shape[1], 3, 2)
            n_iter=3
        cost = layer.output.mean()

        assert cost.type.ndim == 0

        print layer.params

        gparams = theano.tensor.grad(cost, layer.params)
        updates = [(p, p - s_lr*gp) for p, gp in zip(layer.params, gparams)]

        train_nll = pfunc([x, y, s_lr], [], updates=updates)

        xval = theano._asarray(
            rng.uniform(size=(conf.ft_batchsize, x.type.shape[1])),
            dtype=conf.dtype2,
            )
        yval = numpy.arange(conf.ft_batchsize)
        for i in xrange(n_iter):
            train_nll(xval, yval, conf.lr)
