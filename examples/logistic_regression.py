import sys
sys.path.insert(0, '..')
import theano
from theano import tensor as T
from theano.sandbox import nnet_ops
from theano.sandbox import module

import numpy as N


class LogisticRegression(module.FancyModule):
    class __instance_type__(module.FancyModuleInstance):
        def initialize(self, n_in, n_out):
            #self.component is the LogisticRegressionTemplate instance that built this guy.

            self.w = N.random.randn(n_in, n_out)
            self.b = N.random.randn( n_out)
            self.lr = 0.01

    def __init__(self, x = None, targ = None):
        super(LogisticRegression, self).__init__() #boilerplate

        self.x = x if x is not None else T.matrix()
        self.targ = targ if targ is not None else T.lvector()

        self.w = module.Member(T.matrix())   #automatically names
        self.b = module.Member(T.vector())   #automatically names
        self.lr = module.Member(T.dscalar()) #provides an external interface to change it
        #and makes it an implicit input to any Method you build.

        self.params = [self.w, self.b]

        xent, y = nnet_ops.crossentropy_softmax_1hot(
                T.dot(self.x, self.w) + self.b, self.targ)

        gparams = T.grad(xent, self.params)

        self.update = module.Method([self.x, self.targ], xent,
                updates = dict((p, p - self.lr * g) for p, g in zip(self.params, gparams)))
        self.apply = module.Method([self.x], T.argmax(T.dot(self.x, self.w) + self.b, axis=1))

lr = LogisticRegression().make(10, 5, mode='FAST_COMPILE')

data_x = N.random.randn(10,10)
data_y = (N.random.randn(10) > 0)


print lr.params
print lr.w
print lr.b
print lr
for i in xrange(100):
    xe = lr.update(data_x, data_y)
    print N.sum(xe)
    


