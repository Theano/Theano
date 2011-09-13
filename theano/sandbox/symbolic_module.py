import copy, inspect
import theano
import theano.tensor as T

#import klass

def symbolic(f):
    f.__is_symbolic = True
    return f

class InitGraph(type):
    def __init__(cls, name, bases, dct):
        #print 'INITIALIZING', name
        super(InitGraph, cls).__init__(name, bases, dct)
        def just_symbolic(dct):
            def filter(k,v):
                return True
                if getattr(v, '__is_symbolic', False):
                    return True
                if issubclass(v, SymbolicModule):
                    return True
                return isinstance(v, theano.Variable) and not k.startswith('_')
            r = {}
            for key, val in dct.items():
                if filter(key, val):
                    r[key] = val
            return r
        build_graph_rval = cls.build_graph()
        if not isinstance(build_graph_rval, dict):
            raise TypeError('%s.build_graph did not return dictionary' % cls)
        dct = just_symbolic(build_graph_rval)
        for key, val in dct.items():
            #print '  adding class attribute', key
            if isinstance(val, theano.Variable) and val.name is None:
                val.name = key
            if callable(val):
                setattr(cls, key, staticmethod(val))
            else:
                setattr(cls, key, val)

class SymbolicModule(object):
    #installs class attributes from build_graph after declaration
    __metaclass__ = InitGraph

    #if we call this function, it will return a new SymbolicModule
    def __new__(self, **kwargs):
        class SymMod(SymbolicModule):
            @staticmethod
            def build_graph(*bg_args, **bg_kwargs):
                #this one is like self.build_graph,
                #except that the kwargs are automatically inserted
                kwcopy = copy.copy(kwargs)
                kwcopy.update(bg_kwargs)
                return self.build_graph(*bg_args, **kwcopy)
        setattr(SymMod, '__name__', self.__name__ + '_derived')
        return SymMod
    @staticmethod
    def build_graph():
        return {}

def issymbolicmodule(thing):
    try:
        return issubclass(thing, SymbolicModule)
    except Exception:
        return False

def issymbolicmethod(thing):
    return getattr(thing, '__symbolic_method', False)

def symbolic_module(f):
    class SymMod(SymbolicModule):
        build_graph = staticmethod(f)
    return SymMod

def symbolicmethod(f):
    f.__symbolic_method = True
    return f

class CompiledModule(object):
    pass

def compile_fn(f, path_locals, common_inputs):
    (args, vararg, kwarg, default) = inspect.getargspec(f)
    if default:
        #this can be handled correctly, in that default arguments trump path_locals
        raise NotImplementedError()
    #make new inputs for the vars named in args
    # this has the effect of creating new storage for these arguments
    # The common storage doesn't get messed with.
    inputs = [In(path_locals.get(name,name)) for name in args]
    inputs.extend([v for k,v in common_inputs.items() if k not in args])
    outputs = f()
    #print 'inputs', inputs
    #print 'outputs', outputs
    compiled_f = theano.function(inputs, outputs)
    updated = []
    return compiled_f, updated

def compile(smod, initial_values={}):
    """
    :type values: dictionary Variable -> value
    """
    def sym_items(mod):
        for k in mod.__dict__:
            if k in ['__module__', 'build_graph', '__doc__']:
                pass
            else:
                yield k, getattr(mod, k)
    def walker(root):
        def modwalker(path_locals, values):
            for val in values:
                yield path_locals, val
                if isinstance(val, list):
                    for s in modwalker(path_locals, val):
                        yield s
                elif isinstance(val, dict):
                    for s in modwalker(path_locals, val.values()):
                        yield s
                elif issymbolicmodule(val):
                    for s in modwalker(val.__dict__, [v for k,v in sym_items(val)]):
                        yield s
                elif isinstance(val, (basestring, int, float)):
                    pass
                elif isinstance(val, theano.Variable):
                    pass
                elif issymbolicmethod(val):
                    pass
                else :
                    # check for weird objects that we would like to disallow
                    # not all objects can be transfered by the clone mechanism below
                    raise TypeError( (val, type(val), getattr(val,'__name__')))
        for blah in modwalker(root.__dict__, [v for k,v in sym_items(root)]):
            yield blah

    #Locate all the starting nodes, and create containers entries for their values
    inputs = {}
    for path_locals, val in walker(smod):
        if isinstance(val, theano.Variable) and (val.owner is None) and (val not in inputs):
            inputs[val] = theano.In(val, value=theano.gof.Container(val, ['a']))

    assert len(inputs) == len([v for v in inputs.items()])

    #Locate all the functions to compile, and compile them
    compiled_functions = {}
    for path_locals, val in walker(smod):
        if issymbolicmethod(val):
            f, update_expressions = compile_fn(val, path_locals, inputs)
            compiled_functions[val] = f

    #Now replicate the nested structure of the SymbolicModule smod
    #with CompiledModules instead

    reflected = {}
    def reflect(thing):
        #UNHASHABLE TYPES
        if isinstance(thing, list):
            return [reflect(e) for e in thing]
        if isinstance(thing, dict):
            raise NotImplementedError()

        #HASHABLE TYPES
        if thing not in reflected:
            if issymbolicmodule(thing):
                class CMod(CompiledModule):
                    pass
                setattr(CMod, '__name__', thing.__name__ + '_compiled')
                #TODO: consider an instance of the class, or the class itself?
                # which is easier for copying?
                cmod = CMod()
                reflected[thing] = cmod
                for key, val in sym_items(thing):
                    setattr(CMod, key, reflect(val))
            elif isinstance(thing, (basestring, int, float)):
                reflected[thing] = thing
            elif isinstance(thing, theano.Variable):
                if thing.owner is None:
                    def getter(s):
                        return inputs[thing].value.value
                    def setter(s, v):
                        inputs[thing].value.storage[0] = v
                    p = property(getter, setter)
                    print p
                    reflected[thing] = p
                else:
                    reflected[thing] = None #TODO: how to reflect derived resuls?
            elif issymbolicmethod(thing):
                reflected[thing] = compiled_functions[thing]
            else :
                # check for weird objects that we would like to disallow
                # not all objects can be transfered by the clone mechanism below
                raise TypeError('reflecting not supported for',
                        (thing, type(thing), getattr(thing, '__name__', None)))
        return reflected[thing]
    rval = reflect(smod)
    rval.__inputs = inputs
    rval.__compiled_functions = compiled_functions
    return rval

@symbolic_module
def LR(x=None, y=None, v=None, c=None, l2_coef = None):
    x = x if x else T.dmatrix()     #our points, one point per row
    y = y if y else T.dmatrix()     #targets , one per row
    v = v if v else T.dmatrix()     #first layer weights
    c = c if c else T.dvector()     #first layer weights
    l2_coef = l2_coef if l2_coef else T.dscalar()

    pred = T.dot(x, v) + c
    sse = T.sum((pred - y) * (pred - y))
    mse = sse / T.shape(y)[0]
    v_l2 = T.sum(T.sum(v*v))
    loss = mse + l2_coef * v_l2

    @symbolicmethod
    def params():
        return [v, c]

    return locals()

@symbolic_module
def Layer(x=None, w=None, b=None):
    x = x if x else T.dmatrix()     #our points, one point per row
    w = w if w else T.dmatrix()     #first layer weights
    b = b if b else T.dvector()     #first layer bias
    y = T.tanh(T.dot(x, w) + b)
    @symbolicmethod
    def params(): return [w,b]
    return locals()

@symbolic_module
def NNet(x=None, y=None, n_hid_layers=2):
    x = x if x else T.dmatrix()     #our points, one point per row
    y = y if y else T.dmatrix()     #targets , one per row
    layers = []
    _x = x
    for i in xrange(n_hid_layers):
        layers.append(Layer(x=_x))
        _x = layers[-1].y
    classif = LR(x=_x)

    @symbolicmethod
    def params():
        rval = classif.params()
        for l in layers:
            rval.extend(l.params())
        print [id(r) for r in rval]
        return rval

    if 0:
        @symbolicmethod
        def update(x, y):
            pp = params()
            gp = T.grad(classif.loss, pp)
            return dict((p,p - 0.01*g) for p, g in zip(pp, gp))

    return locals()
nnet = compile(NNet)

print nnet
print nnet.params()
print nnet.params.__dict__['finder'][NNet.layers[0].w]
nnet.params[NNet.layers[0].w] = [[6]]
print nnet.params()
print nnet.params()

if 0:
    def deco(f):
        class SymMod(SymbolicModule):
            def __call__(self, *args, **kwargs):
                #return another SymbolicModule built like self
                def dummy(*dargs, **dkwargs):
                    print 'args', args, dargs
                    print 'kwargs', kwargs, dkwargs
                    return f(*args, **kwargs)
                return deco(dummy)

        locals_dict = f()
        for key, val in locals_dict.items():
            if isinstance(val, theano.Variable):
                try:
                    kres = klass.KlassMember(val)
                except Exception:
                    kres = klass.KlassVariable(val)
                setattr(SymMod, key, kres)
            elif callable(val) and getattr(val, '__is_symbolic'):
                setattr(SymMod, key, val)

        return SymMod()

    @deco
    def logistic_regression(
            x=T.dmatrix(),    #our points, one point per row
            y=T.dmatrix(),    #our targets
            v=T.dmatrix(),    #first layer weights
            c=T.dvector(),    #first layer bias
            l2_coef = T.dscalar()
            ):
        pred = T.dot(x, v) + c
        sse = T.sum((pred - y) * (pred - y))
        v_l2 = T.sum(T.sum(v*v))
        loss = sse + l2_coef * v_l2

        @symbolic
        def params(): return [v, c]

        return just_symbolic(locals())

    @deco
    def tanh_layer(
            top_part=None,
            x=T.dmatrix(),    #our points, one point per row
            w=T.dmatrix(),    #first layer weights
            b=T.dvector(),    #first layer bias
            **kwargs #other things from logistic_regression
            ):
        hid = T.tanh(T.dot(x, w) + b)
        if top_part:
            print 'top_part', top_part, 'kwargs', kwargs
            top = top_part(x=hid, **kwargs) # SymbolicModule
            def params(): return top.params() + [w, b]
        else:
            def params(): return [w, b]
        return just_symbolic(locals())

    if 0:
        print 'logistic_regression', logistic_regression
        print 'tanh_layer', tanh_layer
        print 'nnet1', nnet1
    nnet1 = tanh_layer(logistic_regression)
    nnet2 = tanh_layer(nnet1)
    print 'nnet2', nnet2

if 0:
    class SymbolicModule(object):
        name = "__no_name__" #name of this module

        variable_table = {}  #map strings (names) to Variables
        method_table = {}  #map strings to compilable functions
        include_list = []

        constructor_fn = None

        def build(self):
            """Run the body of the included modules in order, using the current variables and imports
            """

        def include(self, symbolic_module, name=None):
            """This redefines the symbols in the kwargs
            """
            name = symbolic_module.name if name is None else name

        def __init__(self, constructor_fn=None):
            """ A constructor fn builds
            - a graph on top of the variable table, and
            - compilable methods.
            """

    @SymbolicModule_fromFn
    def neural_net(
            x=T.dmatrix(),    #our points, one point per row
            y=T.dmatrix(),    #our targets
            w=T.dmatrix(),    #first layer weights
            b=T.dvector(),    #first layer bias
            v=T.dmatrix(),    #second layer weights
            c=T.dvector(),    #second layer bias
            step=T.dscalar(), #step size for gradient descent
            l2_coef=T.dscalar() #l2 regularization amount
            ):
        """Idea A:
        """
        hid = T.tanh(T.dot(x, w) + b)
        pred = T.dot(hid, v) + c
        sse = T.sum((pred - y) * (pred - y))
        w_l2 = T.sum(T.sum(w*w))
        v_l2 = T.sum(T.sum(v*v))
        loss = sse + l2_coef * (w_l2 + v_l2)

        def symbolic_params(cls):
            return [cls.w, cls.b, cls.v, cls.c]

        def update(cls, x, y, **kwargs):
            params = cls.symbolic_params()
            gp = T.grad(cls.loss, params)
            return [], [In(p, update=p - cls.step * g) for p,g in zip(params, gp)]

        def predict(cls, x, **kwargs):
            return cls.pred, []

        return locals()

    #at this point there is a neural_net module all built and compiled,
    # there is also a neural_net.symbolic_module which can be imported.

    @SymbolicModule_fromFn
    def PCA(
            x = T.dmatrix(),
            var_thresh = T.dscalar()
            ):
        #naive version, yes
        s,v,d = T.svd(x)
        acc = T.accumulate(v)
        npc = T.lsearch(acc, var_thresh * T.sum(v))
        y = s[:,:npc]
        #transform will map future points x into the principle components space
        transform = d[:npc,:].T / v[:npc]
        return locals()

    #at this point there is a neural_net module all built and compiled,
    # there is also a neural_net.symbolic_module which can be imported.


    #running this means:
    nnet_on_pca = neural_net(x=PCA.y, submodules=[PCA])
    #nnet_on_pca = SymbolicModule()
    #nnet_on_pca.include(PCA) #an already-instantiated Module
    #nnet_on_pca.x = nnet_on_pca.PCA.y #configure this Module
    #nnet_on_pca.build(neural_net) # instantiate this module

    nnet_on_pca = neural_net(
            substitute=dict(x=PCA.x),
            submodules=[PCA],
            add_symbols=dict(x=PCA.x)
            )

    nnet = logistic_regression(
            redefine={'x':(LogisticLayer.x, LogisticLayer.y)},
            submodule={'hid':LogisticLayer},
            add_symbols={'x':LogisticLayer.x})


    def stats_collector(r, stat_name):
        """stats_collector(nnet_on_pca.x, 'mean')
        """
        return mean_collector(x=r)
