import numpy
import theano
import cPickle
from theano.sandbox.cuda import use, unuse


def test_compile():
    for order in (('gpu', 'cpu'), ('cpu', 'gpu')):
        use(order[0])
        shared_var = theano.shared(numpy.zeros((10, 10),
                                               dtype=theano.config.floatX)
                                   , name="shared_var")
        if order[0] == 'cpu':
            assert not shared_var._isCudaType(), \
                'The input device is set to CPU,' \
                'so the shared variable should be on the CPU.'
        else:
            assert(shared_var._isCudaType()), \
                'The input device is set to GPU,' \
                ' so the shared variable should be on the GPU.'
        with open('shared_var.pickle', 'w') as file:
            cPickle.dump(shared_var, file)
        unuse()

        use(order[1])
        with open('shared_var.pickle', 'r') as file:
            shared_var = cPickle.load(file)
        if order[1] == 'cpu':
            assert(not shared_var._isCudaType()),\
                'The input device is set to CPU,' \
                ' so the shared variable should be on the CPU.'
        else:
            assert(shared_var._isCudaType()),\
                'The input device is set to GPU,' \
                ' so the shared variable should be on the GPU.'
        unuse()