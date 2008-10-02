
import theano
from theano import tensor as T
from theano import gof
from collections import defaultdict
from itertools import chain
from theano.gof.utils import scratchpad
from copy import copy


def join(*args):
    return ".".join(arg for arg in args if arg)
def split(sym, n=-1):
    return sym.split('.', n)



class KlassComponent(object):
    _name = ""
    
    def bind(self, klass, name):
        if self.bound():
            raise Exception("%s is already bound to %s as %s" % (self, self.klass, self.name))
        self.klass = klass
        self.name = join(klass.name, name)

    def bound(self):
        return hasattr(self, 'klass')

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.__class__.__name__

    def __get_name__(self):
        return self._name

    def __set_name__(self, name):
        self._name = name

    name = property(lambda self: self.__get_name__(),
                    lambda self, value: self.__set_name__(value))



class KlassResult(KlassComponent):
    
    def __init__(self, r):
        self.r = r

    def __set_name__(self, name):
        super(KlassResult, self).__set_name__(name)
        self.r.name = name

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, self.r)



class KlassMember(KlassResult):

    def __init__(self, r):
        if r.owner:
            raise ValueError("A KlassMember must not be the result of a previous computation.")
        super(KlassMember, self).__init__(r)



class KlassMethod(KlassComponent):

    def __init__(self, inputs, outputs, updates = {}, **kwupdates):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        self.inputs = inputs
        self.outputs = outputs
        self.updates = dict(updates, **kwupdates)

    def bind(self, klass, name):
        super(KlassMethod, self).bind(klass, name)
        self.inputs = [klass.resolve(i, KlassResult).r for i in self.inputs]
        self.outputs = [klass.resolve(o, KlassResult).r for o in self.outputs] \
            if isinstance(self.outputs, (list, tuple)) \
            else klass.resolve(self.outputs, KlassResult).r
        updates = self.updates
        self.updates = {}
        self.extend(updates)

    def extend(self, updates = {}, **kwupdates):
        if not hasattr(self, 'klass'):
            self.updates.update(updates)
            self.updates.update(kwupdates)
        else:
            for k, v in chain(updates.iteritems(), kwupdates.iteritems()):
                k, v = self.klass.resolve(k, KlassMember), self.klass.resolve(v, KlassResult)
                self.updates[k.r] = v.r

    def __str__(self):
        return "KlassMethod(%s -> %s%s%s)" % \
            (self.inputs,
             self.outputs,
             "; " if self.updates else "",
             ", ".join("%s <= %s" % (old, new) for old, new in self.updates.iteritems()))



class Klass(KlassComponent):

    def __new__(cls, *args, **kwargs):
        self = object.__new__(cls)
        self.__dict__['__components__'] = {}
        self.__dict__['_name'] = ""
        self.__dict__['__components_list__'] = []
        self.__dict__['__component_names__'] = []
        return self

    ###
    ### Access to the klass members and methods
    ###

    def resolve(self, symbol, filter = None):
        if isinstance(symbol, gof.Result):
            if not filter or filter is KlassResult:
                return KlassResult(symbol)
            for component in self.__components_list__:
                if isinstance(component, Klass):
                    try:
                        return component.resolve(symbol, filter)
                    except:
                        continue
                if isinstance(component, KlassResult) and component.r is symbol:
                    if filter and not isinstance(component, filter):
                        raise TypeError('Did not find a %s instance for symbol %s in klass %s (found %s)' 
                                        % (filter.__name__, symbol, self, type(component).__name__))
                    return KlassResult(symbol)
            raise ValueError('%s is not part of this klass or any of its inner klasses. Please add it to the structure before you use it.' % symbol)
        elif isinstance(symbol, str):
            sp = split(symbol, 1)
            if len(sp) == 1:
                try:
                    result = self.__components__[symbol]
                except KeyError:
                    raise AttributeError('Could not resolve symbol %s in klass %s' % (symbol, self))
                if filter and not isinstance(result, filter):
                    raise TypeError('Did not find a %s instance for symbol %s in klass %s (found %s)' 
                                    % (filter.__name__, symbol, self, type(result).__name__))
                return result
            else:
                sp0, spr = sp
                klass = self.__components__[sp0]
                if not isinstance(klass, Klass):
                    raise TypeError('Could not get subattribute %s of %s' % (spr, klass))
                return klass.resolve(spr, filter)
        else:
            raise TypeError('resolve takes a string or Result argument, not %s' % symbol)

    def members(self, as_results = False):
        filtered = [x for x in self.__components_list__ if isinstance(x, KlassMember)]
        if as_results:
            return [x.r for x in filtered]
        else:
            return filtered

    def methods(self):
        filtered = [x for x in self.__components_list__ if isinstance(x, KlassMethod)]
        return filtered

    def member_klasses(self):
        filtered = [x for x in self.__components_list__ if isinstance(x, Klass)]
        return filtered

    ###
    ### Make
    ###

    def __make__(self, mode, stor = None):
        if stor is None:
            stor = scratchpad()
            self.initialize_storage(stor)

        members = []
        methods = []
        rval = KlassInstance()
        for component, name in zip(self.__components_list__, self.__component_names__):
            if isinstance(component, KlassMember):
                container = getattr(stor, name)
                members.append((component, container))
                rval.__finder__[name] = container
            elif isinstance(component, Klass):
                inner, inner_members = component.__make__(mode, getattr(stor, name))
                rval.__dict__[name] = inner
                members += inner_members
            elif isinstance(component, KlassMethod):
                methods.append(component)

        for method in methods:
            inputs = list(method.inputs)
            for (component, container) in members:
                r = component.r
                update = method.updates.get(component.r, component.r)
                inputs.append(theano.In(result = r,
                                        update = update,
                                        value = container,
                                        name = r.name and split(r.name)[-1],
                                        mutable = True,
                                        strict = True))
            fn = theano.function(inputs,
                                 method.outputs,
                                 mode = mode)
            rval.__dict__[split(method.name)[-1]] = fn

        return rval, members

    def make(self, mode = 'FAST_RUN', **init):
        rval = self.__make__(mode)[0]
        self.initialize(rval, **init)
        return rval

    ###
    ### Instance setup and initialization
    ###

    def initialize_storage(self, stor):
        if not hasattr(stor, '__mapping__'):
            stor.__mapping__ = {}
        mapping = stor.__mapping__
        for name, component in self.__components__.iteritems():
            if isinstance(component, Klass):
                sp = scratchpad()
                setattr(stor, name, sp)
                sp.__mapping__ = mapping
                component.initialize_storage(sp)
            elif isinstance(component, KlassMember):
                r = component.r
                if r in mapping:
                    container = mapping[r]
                else:
                    container = gof.Container(r.type,
                                              name = name,
                                              storage = [None])
                    mapping[r] = container
                setattr(stor, name, container)

    def initialize(self, inst, **init):
        for k, v in init.iteritems():
            inst[k] = v

    ###
    ### Magic methods and witchcraft
    ###

    def __setattr__(self, attr, value):
        if attr == 'name':
            self.__set_name__(value)
            return
        elif attr in ['_name', 'klass']:
            self.__dict__[attr] = value
            return
        if isinstance(value, gof.Result):
            value = KlassResult(value)
        if isinstance(value, KlassComponent):
            value.bind(self, attr)
        else:
            self.__dict__[attr] = value
            return
        self.__components__[attr] = value
        self.__components_list__.append(value)
        self.__component_names__.append(attr)
        if isinstance(value, KlassResult):
            value = value.r
        self.__dict__[attr] = value

    def __set_name__(self, name):
        orig = self.name
        super(Klass, self).__set_name__(name)
        for component in self.__components__.itervalues():
            if orig:
                component.name = join(name, component.name[len(orig):])
            else:
                component.name = join(name, component.name)

    def __str__(self):
        n = len(self.name)
        if n: n += 1
        member_names = ", ".join(x.name[n:] for x in self.members())
        if member_names: member_names = "members: " + member_names
        method_names = ", ".join(x.name[n:] for x in self.methods())
        if method_names: method_names = "methods: " + method_names
        klass_names = ", ".join(x.name[n:] for x in self.member_klasses())
        if klass_names: klass_names = "inner: " + klass_names
        return "Klass(%s)" % "; ".join(x for x in [self.name, member_names, method_names, klass_names] if x)



class KlassInstance(object):

    def __init__(self):
        self.__dict__['__finder__'] = {}

    def __getitem__(self, attr):
        if isinstance(attr, str):
            attr = split(attr, 1)
            if len(attr) == 1:
                return self.__finder__[attr[0]].value
            else:
                return getattr(self, attr[0])[attr[1]]
        else:
            raise TypeError('Can only get an item via string format: %s' % attr)

    def __setitem__(self, attr, value):
        if isinstance(attr, str):
            attr = split(attr, 1)
            if len(attr) == 1:
                self.__finder__[attr[0]].value = value
            else:
                getattr(self, attr[0])[attr[1]] = value
        else:
            raise TypeError('Can only set an item via string format: %s' % attr)

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value





from pylearn import nnet_ops as NN
import numpy as N

# class Regression(Klass):

#     def __init__(self, input = None, target = None):

#         if not input:
#             input = T.matrix('input')
#         if not target:
#             target = T.matrix('target')

#         # PARAMETERS
#         self.w = KlassMember(T.matrix())  #the linear transform to apply to our input points
#         self.b = KlassMember(T.vector())  #a vector of biases, which make our transform affine instead of linear

#         # HYPER-PARAMETERS
#         self.l2_coef = KlassMember(T.scalar())
#         self.stepsize = KlassMember(T.scalar())  # a stepsize for gradient descent

#         # REGRESSION MODEL AND COSTS TO MINIMIZE
#         self.prediction = NN.softmax(T.dot(input, self.w) + self.b)
#         self.cross_entropy = -T.sum(target * T.log(self.prediction) + (1 - target) * T.log(1 - self.prediction), axis=1)
#         self.xe_cost = T.sum(self.cross_entropy)
#         self.wreg = self.l2_coef * T.sum(self.w * self.w)
#         self.cost = self.xe_cost + self.wreg

#         # GET THE GRADIENTS NECESSARY TO FIT OUR PARAMETERS
#         self.grad_w, self.grad_b = T.grad(self.cost, [self.w, self.b])

#         self.update = KlassMethod([input, target],
#                                   self.cost,
#                                   w = self.w - self.stepsize * self.grad_w,
#                                   b = self.b - self.stepsize * self.grad_b)

#         self.apply = KlassMethod(input, self.prediction)

#     def initialize(self, obj, input_size = None, target_size = None, **init):
#         if (input_size is None) ^ (target_size is None):
#             raise ValueError("Must specify input_size and target_size or neither.")
#         obj.l2_coef = 0
#         super(Regression, self).initialize(obj, **init)
#         if input_size is not None:
#             obj.w = N.random.uniform(size = (input_size, target_size), low = -0.5, high = 0.5)
#             obj.b = N.zeros(target_size)




class RegressionLayer(Klass):

    def __init__(self, input = None, target = None, regularize = True):

        # MODEL CONFIGURATION
        self.regularize = regularize

        # ACQUIRE/MAKE INPUT AND TARGET
        if not input:
            input = T.matrix('input')
        if not target:
            target = T.matrix('target')

        # HYPER-PARAMETERS
        self.l2_coef = KlassMember(T.scalar())
        self.stepsize = KlassMember(T.scalar())  # a stepsize for gradient descent

        # PARAMETERS
        self.w = KlassMember(T.matrix())  #the linear transform to apply to our input points
        self.b = KlassMember(T.vector())  #a vector of biases, which make our transform affine instead of linear

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
        self.update = KlassMethod([input, target],
                                  self.cost,
                                  w = self.w - self.stepsize * self.grad_w,
                                  b = self.b - self.stepsize * self.grad_b)

        self.apply = KlassMethod(input, self.prediction)

    def params(self):
        return self.w, self.b

    def initialize(self, obj, input_size = None, target_size = None, **init):
        super(RegressionLayer, self).initialize(obj, **init)
        if input_size and target_size:
            sz = (input_size, target_size)
            obj.w = N.random.uniform(size = sz, low = -0.5, high = 0.5)
            obj.b = N.zeros(target_size)

    def build_regularization(self):
        return T.zero() # no regularization!


class SoftmaxXERegression(RegressionLayer):

    def build_prediction(self):
        return NN.softmax(self.activation)

    def build_classification_cost(self, target):
        self.classification_cost_matrix = target * T.log(self.prediction) + (1 - target) * T.log(1 - self.prediction)
        self.classification_costs = -T.sum(self.classification_cost_matrix, axis=1)
        return T.sum(self.classification_costs)

    def build_regularization(self):
        return self.l2_coef * T.sum(self.w * self.w)


#softmax_xe_regression = RegressionLayer(NN.softmax, xe)


class AutoEncoder(Klass):

    def __init__(self, input = None, regularize = True, tie_weights = True):

        # MODEL CONFIGURATION
        self.regularize = regularize
        self.tie_weights = tie_weights

        # ACQUIRE/MAKE INPUT
        if not input:
            input = T.matrix('input')

        # HYPER-PARAMETERS
        self.stepsize = KlassMember(T.scalar())
        self.l2_coef = KlassMember(T.scalar())

        # PARAMETERS
        self.w1 = KlassMember(T.matrix())
        if not tie_weights:
            self.w2 = KlassMember(T.matrix())
        else:
            self.w2 = self.w1.T
        self.b1 = KlassMember(T.vector())
        self.b2 = KlassMember(T.vector())

        # HIDDEN LAYER
        self.hidden_activation = T.dot(input, self.w1) + self.b1
        self.hidden = self.build_hidden()

        # RECONSTRUCTION LAYER
        self.output_activation = T.dot(self.hidden, self.w2) + self.b2
        self.output = self.build_output()

        # RECONSTRUCTION COST
        self.reconstruction_cost = self.build_reconstruction_cost(input)

        # REGULARIZATION COST
        self.regularization = self.build_regularization()

        # TOTAL COST
        self.cost = self.reconstruction_cost
        if self.regularize:
            self.cost = self.cost + self.regularization
        
        # GRADIENTS AND UPDATES
        params = self.params()
        gradients = T.grad(self.cost, params)
        updates = dict((p, p - self.stepsize * g) for p, g in zip(params, gradients))

        # INTERFACE METHODS
        self.update = KlassMethod(input, self.cost, updates)
        self.reconstruction = KlassMethod(input, self.output)
        self.representation = KlassMethod(input, self.hidden)

    def params(self):
        if self.tie_weights:
            return self.w1, self.b1, self.b2
        else:
            return self.w1, self.w2, self.b1, self.b2

    def initialize(self, obj, input_size = None, hidden_size = None, **init):
        if (input_size is None) ^ (hidden_size is None):
            raise ValueError("Must specify hidden_size and target_size or neither.")
        obj.l2_coef = 0
        super(AutoEncoder, self).initialize(obj, **init)
        if input_size is not None:
            sz = (input_size, hidden_size)
            obj.w1 = N.random.uniform(size = sz, low = -0.5, high = 0.5)
            if not self.tie_weights:
                obj.w2 = N.random.uniform(size = list(reversed(sz)), low = -0.5, high = 0.5)
            obj.b1 = N.zeros(hidden_size)
            obj.b2 = N.zeros(input_size)

    def build_regularization(self):
        return T.zero() # no regularization!


class SigmoidXEAutoEncoder(AutoEncoder):

    def build_hidden(self):
        return NN.sigmoid(self.hidden_activation)

    def build_output(self):
        return NN.sigmoid(self.output_activation)

    def build_reconstruction_cost(self, input):
        self.reconstruction_cost_matrix = input * T.log(self.output) + (1 - input) * T.log(1 - self.output)
        self.reconstruction_costs = -T.sum(self.reconstruction_cost_matrix, axis=1)
        return T.sum(self.reconstruction_costs)

    def build_regularization(self):
        if self.tie_weights:
            return self.l2_coef * T.sum(self.w1 * self.w1)
        else:
            return self.l2_coef * T.sum(self.w1 * self.w1) + T.sum(self.w2 * self.w2)


class Stacker(Klass):

    def __init__(self, metaklasses, input = None, target = None, regularize = False):
        current = input
        self.layers = []
        for i, (metaklass, outname) in enumerate(metaklasses):
            layer = metaklass(current, regularize = regularize)
            self.layers.append(layer)
            setattr(self, "layer%i" % (i + 1), layer)
            current = getattr(current, outname)
            
        self.output = current
        self.classification_cost = self.build_classification_cost()
        self.regularization = self.build_regularization()
        self.cost = self.classification_cost
        if regularize:
            self.cost = self.cost + self.regularization
        params = self.params()
        gradients = T.grad(self.cost, params)
        updates = dict((p, p - self.stepsize * g) for p, g in zip(params, gradients))

        # INTERFACE METHODS
        self.update = KlassMethod(input, self.cost, updates)
        self.compute = KlassMethod(input, self.output)



# r = SoftmaxXERegression(regularize = False)
# o = r.make(mode = 'FAST_RUN',
#            input_size = 4,
#            target_size = 2,
#            stepsize = 0.1)

# inputs = N.asarray([[x%2,(x>>1)%2,(x>>2)%2,(x>>3)%2] for x in xrange(16)])
# targets = N.asarray([[1, 0] if (x>>1)%2 else [0, 1] for x in xrange(16)])


# print o.w
# for i in xrange(100):
#     o.update(inputs, targets)
# print N.hstack([targets, o.apply(inputs)]).round()



# aa = SigmoidXEAutoEncoder(tie_weights = True)
# o = aa.make(mode = 'FAST_RUN',
#             input_size = 4,
#             hidden_size = 2,
#             stepsize = 0.1)

# inputs = N.asarray([[x%2,(x>>1)%2,(x>>2)%2,(x>>3)%2] for x in xrange(16) if x % 2])

# print o.w1
# #print o.w2

# for i in xrange(1000):
#     o.update(inputs)

# print N.hstack([inputs, o.reconstruction(inputs)]).round()

# print o.representation(inputs)







# def make_incdec_klass():
#     k = Klass()
#     n = T.scalar('n')
#     k.c = KlassMember(T.scalar()) # state variables must be wrapped with KlassMember
#     k.inc = KlassMethod(n, [], c = k.c + n) # k.c <= k.c + n
#     k.dec = KlassMethod(n, [], c = k.c - n) # k.c <= k.c - n
#     k.plus10 = KlassMethod([], k.c + 10) # k.c is always accessible since it is a member of this klass
#     return k


# k = Klass()
# k.incdec1 = make_incdec_klass()
# k.incdec2 = make_incdec_klass()
# k.sum = KlassMethod([], k.incdec1.c + k.incdec2.c)
# inst = k.make(**{'incdec1.c': 0, 'incdec2.c': 0}) # I'm considering allowing k.make(incdec1__c = 0, incdec2__c = 0)... thoughts?
# inst.incdec1.inc(2)
# inst.incdec1.dec(4)
# inst.incdec2.inc(6)
# assert inst.incdec1.c == -2 and inst.incdec2.c == 6
# assert inst.sum() == 4 # -2 + 6

# print inst.sum(), inst.incdec1.c, inst.incdec2.c








# k = Klass()
# k.x, k.y = T.scalars('xy')
# k.z = k.x + k.y
# k.s = KlassMember(T.scalar())
# k.f = KlassMethod(['x', 'y'], 'z', s = k.x)


# k2 = Klass()
# k2.paf = k
# k2.x, k2.y = T.scalars('ab')
# k2.z = k2.x + k2.y + k.s
# k2.t = KlassMember(T.scalar())
# k2.f = KlassMethod(['x', 'y'], k2.z, {k2.t: k2.t + 3, k.s: k.s + 5})


# obj = k2.make(**{'paf.s': 2, 't': 3})

# print obj.t, obj.paf.s
# print obj.f(7, 8)
# print obj.t, obj.paf.s
# print obj.paf.f(1, 2)
# print obj.t, obj.paf.s
# print obj['paf.s']
# print obj[k2.paf.s]



















# class AutoEncoder(Klass):
    
#     def __init__(self, activation_function):
#         self.activation_function = activation_function

#     def build(__self, input):
#         self = copy(__self)
#         self.input = input
#         self.W1, self.W2 = T.matrices(2)
#         self.b1, self.b2 = T.vectors(2)
#         self.lr = T.scalar()
#         return self

#     def initialize(self, nhid...):
#         pass


# class Stacker(Klass):

#     def build(self, input, target, *builders):
#         self.input, self.target = input, target
#         self.lr = T.scalar()
#         current = self.input
#         layers = []
#         for i, builder in enumerate(builders):
#             layer = builder(current)
#             layers.append(layer)
#             setattr(self, 'layer%i' % (i+1), layer)
#             current = layer.hidden
#         self.output = current.output
#         self.update = KlassMethod(['input', 'target'], 'cost')
        

# model = Stacker(AutoEncoder, AutoEncoder, NNLayer)

# model.var = T.mean(T.sqr(model.costs)) - T.sqr(model.cost)
# model.variance = KlassMethod(['input', 'target'], 'var')

# model.var_stor = T.scalar()
# model.update.extend(var_stor = model.var)





# class Stacked(Klass):

#     def __init__(self, x, y, stepsize):
#         lay1 = Regression(x, y)
#         lay2 = Regression(lay1.interesting_representation, y)
        
#         cost1 = lay1.cost + lay2coef * lay2.cost 

#         cost2 = lay2.cost

#         cost3 = rbm_interpreation_cost(lay2)

          

# T.sum(lay2.cross_entropy) + l2_coef * (T.sum(T.sum(w1*w1)) + T.sum(T.sum(w2*w2)))
        

#         self.update = KlassMethod([x, y, stepsize],
#                                   lay2.cost,
#                                   **{lay1.w: lay1.w - stepsize * grad_w1,
#                                      etc})

#     def initialize_storage(self, stor):
#         stor = super().initialize_storage(stor)
#         stor.lay1.b = stor.lay2.b

         
#     def _instance_print_w(self):
#         print self.w.value


# class LinReg(object):
#     __metaklass__  = Stacked

#     def __init__(self, x, y):

#         #make...  initialize, allocate... blah blah blah

#     def print_w(self):
        



#     #
#     # GET THE GRADIENTS NECESSARY TO FIT OUR PARAMETERS

#     update_fn = theano.function(
#         inputs = [x, y, stepsize,
#             In(w,
#                 name='w',
#                 value=numpy.zeros((n_in, n_out)),
#                 update=w - stepsize * grad_w,
#                 mutable=True,
#                 strict=True)
#             In(b,
#                 name='b',
#                 value=numpy.zeros(n_out),
#                 update=b - lr * grad_b,
#                 mutable=True,
#                 strict=True)
#         ],
#         outputs = cost,
#         mode = 'EXPENSIVE_OPTIMIZATIONS')

#     apply_fn = theano.function(
#         inputs = [x, In(w, value=update_fn.storage[w]), In(b, value=update_fn.storage[b])],
#         outputs = [prediction])

#     return update_fn, apply_fn









# class AutoEncoder(Klass):

# #     def __init__(self, activation_function, cost_function, tie_weights = True):
# #         self.activation_function = activation_function
# #         self.cost_function = cost_function
# #         self.tie_weights = tie_weights

#     def __init__(self, input = None, tie_weights = True):
#         self.tie_weights = tie_weights
#         if not input:
#             input = T.matrix('input')

#         self.stepsize = KlassMember(T.scalar())
#         self.l2_coef = KlassMember(T.scalar())

#         self.code = SoftmaxXERegression(input) #RegressionLayer(self.activation_function, self.cost_function).build(input)
#         self.hidden = self.code.prediction
#         self.decode = SoftmaxXERegression(self.hidden, transpose_weights = True) #RegressionLayer(self.activation_function, self.cost_function, code.w.T).build(self.hidden)

#         self.rec = self.decode.prediction
#         self.build_classification_cost(input)

#         self.grad_w1, self.grad_w2, self.grad_b1, self.grad_b2 = \
#             T.grad(self.cost, [self.code.w, self.decode.w, self.code.b, self.decode.b])

#         if self.tie_weights:
#             self.update = KlassMethod(input,
#                                       self.cost,
#                                       {self.code.w: self.code.w - self.stepsize * (self.grad_w1 + self.grad_w2),
#                                        self.code.b: self.code.b - self.stepsize * self.grad_b1,
#                                        self.decode.b: self.decode.b - self.stepsize * self.grad_b2})
#         else:
#             self.update = KlassMethod(input,
#                                       self.cost,
#                                       {self.code.w: self.code.w - self.stepsize * self.grad_w1,
#                                        self.code.b: self.code.b - self.stepsize * self.grad_b1,
#                                        self.decode.w: self.decode.w - self.stepsize * self.grad_w2,
#                                        self.decode.b: self.decode.b - self.stepsize * self.grad_b2})

#         self.reconstruction = KlassMethod(input, self.rec)
#         self.representation = KlassMethod(input, self.hidden)
#         return self

#     def initialize_storage(self, stor):
#         super(AutoEncoder, self).initialize_storage(stor)
#         if self.tie_weights:
#             stor.decode.w = stor.code.w

#     def initialize(self, obj, input_size = None, hidden_size = None, **init):
#         if (input_size is None) ^ (hidden_size is None):
#             raise ValueError("Must specify input_size and hidden_size or neither.")
#         obj.l2_coef = 0
#         super(AutoEncoder, self).initialize(obj, **init)
#         if input_size is not None:
#             reg = RegressionLayer(self.activation_function, self.cost_function)
#             reg.initialize(obj.code, input_size, hidden_size, stepsize = obj.stepsize)
#             reg2 = RegressionLayer(self.activation_function, self.cost_function, transpose_weights = True)
#             reg2.initialize(obj.decode, hidden_size, input_size, stepsize = obj.stepsize)

