from theano import Type, Variable, Constant


class CVMType(Type):
    def __init__(self, name=None):
        self.ndim = 0
        self.broadcastable = ()
        self.name = name

    def filter(val, strict=False, allow_downcast=None):
        if not self.is_valid_value(val):
            raise ValueError("only works with CVM instances")
        return val

    def is_valid_value(self, value):
        # if this import fails, the type is unusable anyway
        from theano.gof.vm import CVM
        return isinstance(val, CVM)

    @staticmethod
    def values_eq(a, b):
        return id(a) == id(b)

    def make_variable(self, name=None):
        return self.Variable(self, name=name)

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return "CVMType"

    def c_declare(self, name, sub, check_input=True):
        return "CLazyLinker *%(name)s;" % dict(name=name)

    def c_init(self, name, sub):
        return "%(name)s = NULL;" % dict(name=name)

    def c_extract(self, name, sub, check_inputs=True):
        res = ""
        if check_inputs:
            res += """
        if (py_%(name)s == Py_None) {
          PyErr_SetString(PyExc_ValueError, "None instead of CVM");
          %(fail)s
        }
        """
        res += """
        %(name)s = (CLazyLinker *)py_%(name)s;
        Py_INCREF(%(name)s);
        """
        return res % dict(name=name, fail=sub['fail'])

    def c_cleanup(self, name, sub):
        return "Py_XDECREF(%(name)s); %(name)s = NULL;" % dict(name=name)

    def c_sync(self, name, sub):
        return """
        if (!%(name)s) {
            Py_XDECREF(py_%(name)s);
            Py_INCREF(Py_None);
            py_%(name)s = Py_None;
        } else if ((void *)py_%(name)s != (void *)%(name)s) {
            Py_XDECREF(py_%(name)s);
            py_%(name)s = (PyObject *)%(name)s;
            Py_INCREF(py_%(name)s);
        }
        """ % {'name': name}

    def c_headers(self):
        return ['"lazylinker_c.h"']

    def c_header_dirs(self):
        return [os.path.join(theano.__path__[0], 'gof')]

    def c_code_cache_version(self):
        # don't cache for now
        return
        return (0,)


class CVMVariable(Variable):
    pass


CVMType.Variable = CVMVariable


class CVMSignature(object):
    def __init__(self, cvm):
        self.cvm = cvm

    def __eq__(self, other):
        return (type(self) == type(other) and
                id(self.cvm) == id(other.cvm))

    def __hash__(self):
        return hash(type(self)) ^ hash(id(self.cvm))

    def theano_hash(self):
        return hash(self)


class CVMConstant(Constant):
    def signature(self):
        return CVMSignature(self.data)

    def __str__(self):
        return "CVMConstant{%x}" % (id(self.data),)


CVMType.Constant = CVMConstant


class StorageType(Type):
    def __init__(self, name=None):
        self.ndim = 0
        self.broadcastable = ()
        self.name = name

    def filter(val, strict=False, allow_downcast=None):
        if not isintance(val, list):
            raise ValueError("not a list")
        if not all(isinstance(vv, list) and len(vv) == 1
                   for vv in val):
            raise ValueError("not a list of cells")
        return vv

    def __str__(self):
        return "StorageType"

    @staticmethod
    def values_eq(a, b):
        return id(a) == id(b)

    def make_variable(self, name=None):
        return self.Variable(self, name=name)

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def c_dclare(self, name, sub, check_input=True):
        return "PyListObject *%(name)s;" % dict(name=name)

    def c_init(self, name, sub):
        return "%(name)s = NULL;" % dict(name=name)

    def c_extract(self, name, sub, check_input=True):
        return """
        if (py_%(name)s == Py_None) {
            PyErr_SetString(PyExc_ValueError, "expected a list, not None");
            %(fail)s
        }
        %(name)s = (PyListObject *)py_%(name)s;
        Py_INCREF(%(name)s);
        """ % dict(name=name, fail=sub['fail'])

    def c_cleanup(self, name, sub):
        return "Py_XDECREF(%(name)s); %(name)s = NULL;" % dict(name=name)

    def c_sync(self, name, sub):
        return """
        if (!%(name)s) {
            Py_XDECREF(py_%(name)s);
            Py_INCREF(Py_None);
            py_%(name)s = Py_None;
        } else if ((void *)py_%(name)s != (void *)%(name)s) {
            Py_XDECREF(py_%(name)s);
            py_%(name)s = (PyObject *)%(name)s;
            Py_INCREF(py_%(name)s);
        }
        """ % dict(name=name)


    def c_code_cache_version(self):
        # don't cache for now
        return
        return (0,)


class StorageVariable(Variable):
    pass


StorageType.Variable = StorageVariable


class StorageSignature(object):
    def __init__(self, st):
        self.st = st

    def __eq__(self, other):
        return (type(self) == type(other) and
                id(self.st) == id(other.st))

    def __hash__(self):
        return hash(type(self)) ^ hash(id(self.st))

    def theano_hash(self):
        return hash(self)


class StorageConstant(Constant):
    def signature(self):
        return StorageSignature(self.data)

    def __str__(self):
        return "StorageConstant{%x}" % (id(self.data),)


StorageType.Constant = StorageConstant
