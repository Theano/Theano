import theano
from theano import Type
from theano import scalar

class CLArrayType(Type):
    typenum = 11
    dtype = 'float32'
    Variable = None
    Constant = None
    SharedVariable = None
    
    def __init__(self, broadcastable, name=None, dtype=None):
        if dtype != None and dtype != 'float32':
            raise TypeError(self.__class__.__name__+' only supports float32 for now.  Tried using dtype %s for variable %s' % (dtype, name))
        self.broadcastable = tuple(broacastable)
        self.name = name

    def filter(self, data, strict=False, allow_downcast=None):
        return self.filter_inplace(data, None, strict=strict, allow_downcast=allow_downcast)

    def filter_inplace(self, data, old_data, strict=None, allow_downcast=None):
        from basic_ops import as_cl_array
        
        if strict or isinstance(data, cl.array.Array):
            conf.filter_array(data, self.broadcastable)
            return data

        else:
            if not isinstance(data, numpy.ndarray):
                converted_data = theano._asarray(data, self.dtype)
                if allow_downcast is None and type(data) is float and self.dtype == theano.config.floatX:
                    allow_downcast = True
                elif not all(data == converted_data).all():
                    raise TypeError('Will not downcast')
                data = converted_data

            up_dtype = scalar.upcast(self.dtype, data.dtype)
            if up_dtype == self.dtype or allow_downcast:
                res = as_cl_array(data).astype(self.dtype)
                conf.filter_array(res, self.broacastable)
                return res
            else:
                raise TypeError('Will not downcast')

    @staticmethod
    def values_eq(a, b):
        return tensor.TensorType.values_eq(a.get(), b.get())

    @staticmethod
    def values_eq_approx(a, b, allow_remove_inf=False):
        return tensor.TensorType.values_eq_approx(a.get(), b.get(),
                                            allow_remove_inf=allow_remove_inf)
    @staticmethod
    def may_share_memory(a,b):
        # Yes, I am lazy
        return True

    def __eq__(self, other):
        return type(self) == type(other) and other.broadcastable == self.broadcastable

    def __hash__(self):
        return hash(type(self)) ^ hash(self.broadcastable)

    ndim = property(lambda self: len(self.broadcastable), doc = "number of dimensions")
    
    def make_variable(self, name=None):
        return self.Variable(self, name=name)
    
        def __str__(self):
            if self.name:
                return self.name
            else:
                b = self.broadcastable

            if not numpy.any(b):
                s = "%iD" % len(b)
            else:
                s = str(b)

            bcast = {(): 'scalar',
                     (False,): 'vector',
                     (False, True): 'col',
                     (True, False): 'row',
                     (False, False): 'matrix'}.get(b, s)

            return "CLArrayType(%s, %s)" % (str(self.dtype), bcast)
