import unittest

import theano
import numpy as N
from theano import tensor as T
from theano.tensor import nnet as NN
from theano.compile import module as M

class Blah(M.ModuleInstance):
#        self.component #refer the Module
#    def __init__(self, input = None, target = None, regularize = True):
#        super(Blah, self)
    def initialize(self,input_size = None, target_size = None, seed = 1827, 
                   **init):
        if input_size and target_size:
            # initialize w and b in a special way using input_size and target_size
            sz = (input_size, target_size)
            rng = N.random.RandomState(seed)
            self.w = rng.uniform(size = sz, low = -0.5, high = 0.5)
            self.b = N.zeros(target_size)
            self.stepsize = 0.01

    def __eq__(self, other):
        if not isinstance(other.component, SoftmaxXERegression1) and not isinstance(other.component, SoftmaxXERegression2):
            raise NotImplemented
        #we compare the member.
        if (self.w==other.w).all() and (self.b==other.b).all() and self.stepsize == other.stepsize:
            return True
        return False
    def __hash__(self):
        raise NotImplemented

    def fit(self, train, test):
        pass

class RegressionLayer1(M.Module):
    InstanceType=Blah
    def __init__(self, input = None, target = None, regularize = True):
        super(RegressionLayer1, self).__init__() #boilerplate
        # MODEL CONFIGURATION
        self.regularize = regularize
        # ACQUIRE/MAKE INPUT AND TARGET
        if not input:
            input = T.matrix('input')
        if not target:
            target = T.matrix('target')
        # HYPER-PARAMETERS
        self.stepsize = M.Member(T.scalar())  # a stepsize for gradient descent
        # PARAMETERS
        self.w = M.Member(T.matrix())  #the linear transform to apply to our input points
        self.b = M.Member(T.vector())  #a vector of biases, which make our transform affine instead of linear
        # REGRESSION MODEL
        self.activation = T.dot(input, self.w) + self.b
        self.prediction = self.build_prediction()
        # CLASSIFICATION COST
        self.classification_cost = self.build_classification_cost(target)
        # REGULARIZATION COST
        self.regularization = self.build_regularization()
        # TOTAL COST
        self.cost = self.classification_cost
        if self.regularize:
            self.cost = self.cost + self.regularization
        # GET THE GRADIENTS NECESSARY TO FIT OUR PARAMETERS
        self.grad_w, self.grad_b = T.grad(self.cost, [self.w, self.b])
        # INTERFACE METHODS
        self.update = M.Method([input, target],
                                  self.cost,
                                  w = self.w - self.stepsize * self.grad_w,
                                  b = self.b - self.stepsize * self.grad_b)
        self.apply = M.Method(input, self.prediction)
    def params(self):
        return self.w, self.b
    def build_regularization(self):
        return T.zero() # no regularization!

class RegressionLayer2(M.Module):
    def __init__(self, input = None, target = None, regularize = True):
        super(RegressionLayer2, self).__init__() #boilerplate
        # MODEL CONFIGURATION
        self.regularize = regularize
        # ACQUIRE/MAKE INPUT AND TARGET
        if not input:
            input = T.matrix('input')
        if not target:
            target = T.matrix('target')
        # HYPER-PARAMETERS
        self.stepsize = M.Member(T.scalar())  # a stepsize for gradient descent
        # PARAMETERS
        self.w = M.Member(T.matrix())  #the linear transform to apply to our input points
        self.b = M.Member(T.vector())  #a vector of biases, which make our transform affine instead of linear
        # REGRESSION MODEL
        self.activation = T.dot(input, self.w) + self.b
        self.prediction = self.build_prediction()
        # CLASSIFICATION COST
        self.classification_cost = self.build_classification_cost(target)
        # REGULARIZATION COST
        self.regularization = self.build_regularization()
        # TOTAL COST
        self.cost = self.classification_cost
        if self.regularize:
            self.cost = self.cost + self.regularization
        # GET THE GRADIENTS NECESSARY TO FIT OUR PARAMETERS
        self.grad_w, self.grad_b = T.grad(self.cost, [self.w, self.b])
        # INTERFACE METHODS
        self.update = M.Method([input, target],
                                  self.cost,
                                  w = self.w - self.stepsize * self.grad_w,
                                  b = self.b - self.stepsize * self.grad_b)
        self.apply = M.Method(input, self.prediction)
    def params(self):
        return self.w, self.b
    def _instance_initialize(self, obj, input_size = None, target_size = None, 
                             seed = 1827, **init):
        # obj is an "instance" of this module holding values for each member and
        # functions for each method
        #super(RegressionLayer, self).initialize(obj, **init)

        # here we call the superclass's initialize method, which takes all the name: value
        # pairs in init and sets the property with that name to the provided value
        # this covers setting stepsize, l2_coef; w and b can be set that way too
        if input_size and target_size:
            # initialize w and b in a special way using input_size and target_size
            sz = (input_size, target_size)
            rng = N.random.RandomState(seed)
            obj.w = rng.uniform(size = sz, low = -0.5, high = 0.5)
            obj.b = N.zeros(target_size)
            obj.stepsize = 0.01
    def build_regularization(self):
        return T.zero() # no regularization!

class SoftmaxXERegression1(RegressionLayer1):
    """ XE mean cross entropy"""
    def build_prediction(self):
        return NN.softmax(self.activation)
    def build_classification_cost(self, target):
        #self.classification_cost_matrix = target * T.log(self.prediction) + (1 - target) * T.log(1 - self.prediction)
        self.classification_cost_matrix = (target - self.prediction)**2
        self.classification_costs = -T.sum(self.classification_cost_matrix, axis=1)
        return T.sum(self.classification_costs)
    def build_regularization(self):
        self.l2_coef = M.Member(T.scalar()) # we can add a hyper parameter if we need to
        return self.l2_coef * T.sum(self.w * self.w)


class SoftmaxXERegression2(RegressionLayer2):
    """ XE mean cross entropy"""
    def build_prediction(self):
        return NN.softmax(self.activation)
    def build_classification_cost(self, target):
        #self.classification_cost_matrix = target * T.log(self.prediction) + (1 - target) * T.log(1 - self.prediction)
        self.classification_cost_matrix = (target - self.prediction)**2
        self.classification_costs = -T.sum(self.classification_cost_matrix, axis=1)
        return T.sum(self.classification_costs)
    def build_regularization(self):
        self.l2_coef = M.Member(T.scalar()) # we can add a hyper parameter if we need to
        return self.l2_coef * T.sum(self.w * self.w)


class T_function_module(unittest.TestCase):
    def test_Klass_basic_example1(self):
        n, c = T.scalars('nc')
        inc = theano.function([n, ((c, c + n), 0)], [])
        dec = theano.function([n, ((c, c - n), inc.container[c])], []) # we need to pass inc's container in order to share
        plus10 = theano.function([(c, inc.container[c])], c + 10)
        assert inc[c] == 0
        inc(2)
        assert inc[c] == 2 and dec[c] == inc[c]
        dec(3)
        assert inc[c] == -1 and dec[c] == inc[c]
        assert plus10() == 9

    def test_Klass_basic_example2(self):
        m = M.Module()
        n = T.scalar('n')
        m.c = M.Member(T.scalar()) # state variables must be wrapped with ModuleMember
        m.inc = M.Method(n, [], c = m.c + n) # m.c <= m.c + n
        m.dec = M.Method(n, [], c = m.c - n) # k.c <= k.c - n
        m.dec = M.Method(n, [], updates = {m.c: m.c - n})
        #m.dec = M.Method(n, [], updates = {c: m.c - n})#global c don't exist
        #m.dec = M.Method(n, [], m.c = m.c - n) #python don't suppor this syntax
        m.plus10 = M.Method([], m.c + 10) # m.c is always accessible since it is a member of this mlass
        inst = m.make(c = 0) # here, we make an "instance" of the module with c initialized to 0
        assert inst.c == 0
        inst.inc(2)
        assert inst.c == 2
        inst.dec(3)
        assert inst.c == -1
        assert inst.plus10() == 9

    def test_Klass_nesting_example1(self):
        def make_incdec_function():
            n, c = T.scalars('nc')
            inc = theano.function([n, ((c, c + n), 0)], [])
            dec = theano.function([n, ((c, c - n), inc.container[c])], [])
            return inc,dec


        inc1, dec1 = make_incdec_function()
        inc2, dec2 = make_incdec_function()
        a, b = T.scalars('ab')
        sum = theano.function([(a, inc1.container['c']), (b, inc2.container['c'])], a + b)
        inc1(2)
        dec1(4)
        inc2(6)
        assert inc1['c'] == -2 and inc2['c'] == 6
        assert sum() == 4 # -2 + 6

    def test_Klass_nesting_example2(self):
        def make_incdec_module():
            m = M.Module()
            n = T.scalar('n')
            m.c = M.Member(T.scalar()) # state variables must be wrapped with ModuleMember
            m.inc = M.Method(n, [], c = m.c + n) # m.c <= m.c + n
            m.dec = M.Method(n, [], c = m.c - n) # k.c <= k.c - n
            return m

        m = M.Module()
        m.incdec1 = make_incdec_module()
        m.incdec2 = make_incdec_module()
        m.sum = M.Method([], m.incdec1.c + m.incdec2.c)
        inst = m.make(incdec1 = dict(c=0), incdec2 = dict(c=0))
        inst.incdec1.inc(2)
        inst.incdec1.dec(4)
        inst.incdec2.inc(6)
        assert inst.incdec1.c == -2 and inst.incdec2.c == 6
        assert inst.sum() == 4 # -2 + 6

    def test_Klass_Advanced_example(self):
        data_x = N.random.randn(4, 10)
        data_y = [ [int(x)] for x in N.random.randn(4) > 0]
#        print data_x
#        print
#        print data_y
        def test(model):
            model = model.make(input_size = 10,
                                      target_size = 1,
                                      stepsize = 0.1)
            for i in xrange(1000):
                xe = model.update(data_x, data_y)
                if i % 100 == 0:
                    print i, xe
                    pass
            #for inputs, targets in my_training_set():
                #print "cost:", model.update(inputs, targets)


            print "final weights:", model.w
            print "final biases:", model.b

            #Print "some prediction:", model.prediction(some_inputs)
            return model
        m1=test(SoftmaxXERegression1(regularize = False))
        m2=test(SoftmaxXERegression2(regularize = False))
        print "m1",m1 
        print "m2",m2
        print m2==m1
        print m1==m2
        assert m2==m1 and m1==m2

    def test_Klass_extending_klass_methods(self):
        model_module = SoftmaxXERegression1(regularize = False)
        model_module.sum = M.Member(T.scalar()) # we add a module member to hold the sum
        model_module.update.updates.update(sum = model_module.sum + model_module.cost) # now update will also update sum!

        model = model_module.make(input_size = 4,
                                 target_size = 2,
                                 stepsize = 0.1,
                                 sum = 0) # we mustn't forget to initialize the sum

        test = model.update([[0,0,1,0]], [[0,1]]) 
        test += model.update([[0,1,0,0]], [[1,0]])
        assert model.sum == test



        def make_incdec_function():
            n, c = T.scalars('nc')
            inc = theano.function([n, ((c, c + n), 0)], [])
            dec = theano.function([n, ((c, c - n), inc.container[c])], [])
            return inc,dec


        inc1, dec1 = make_incdec_function()
        inc2, dec2 = make_incdec_function()
        a, b = T.scalars('ab')
        sum = theano.function([(a, inc1.container['c']), (b, inc2.container['c'])], a + b)
        inc1(2)
        dec1(4)
        inc2(6)
        assert inc1['c'] == -2 and inc2['c'] == 6
        assert sum() == 4 # -2 + 6

    def test_Klass_basic_example2_more(self):
        m = M.Module()
        m2 = M.Module()
        m2.name="m2" # for better error
        #top level don't have name, but other have auto name.
        
        n = T.scalar('n')
        m.c = M.Member(T.scalar()) # state variables must be wrapped with ModuleMember
        m2.c = M.Member(T.scalar()) # state variables must be wrapped with ModuleMember
        m.dec = M.Method(n, [], c = m.c - n)
        m.inc = M.Method(n, [], c = m.c + n) # m.c <= m.c + n
#        m.inc = M.Method(n, [], c = c + n)#fail c not defined
#syntax error
#        m.inc = M.Method(n, [], m.c = m.c + n)#fail
        m.inc = M.Method(n, [], updates={m.c: m.c + n})
#        m.inc = M.Method(n, [], updates={c: m.c + n})#fail with NameError
#        m.inc = M.Method(n, [], updates={m.c: c + n})#fail with NameError
#        m.inc = M.Method(n, [], updates={c: c + n})#fail with NameError

        m.inc = M.Method(n, [], updates={m.c: m2.c + n})#work! should be allowed?
        a = M.Module()
        a.m1 = m
        a.m2 = m2
        a.make()#should work.
#        self.assertRaises(m.make(c = 0), Error)
        m.inc = M.Method(n, [], updates={m2.c: m.c + n})#work! should be allowed?
#        self.assertRaises(m.make(c = 0), Error)
#        m.inc = M.Method(n, [], updates={m.c: m2.c + m.c+ n})#work! should be allowed?
        m2.inc = M.Method(n, [], updates={m2.c: m2.c + 2*m.c+ n})#work! should be allowed?
#        self.assertRaises(m.make(c = 0), Error)


if __name__ == '__main__':

    if 0:
        unittest.main()
    elif 1:
        module = __import__("test_wiki")
        tests = unittest.TestLoader().loadTestsFromModule(module)
        tests.debug()
    else:
        testcases = []
        testcases.append(T_function_module)

        #<testsuite boilerplate>
        testloader = unittest.TestLoader()
        suite = unittest.TestSuite()
        for testcase in testcases:
            suite.addTest(testloader.loadTestsFromTestCase(testcase))
        unittest.TextTestRunner(verbosity=2).run(suite)
        #</boilerplate>
