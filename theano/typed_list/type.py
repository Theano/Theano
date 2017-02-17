from __future__ import absolute_import, print_function, division
from theano import gof


class TypedListType(gof.Type):
    """

    Parameters
    ----------
    ttype
        Type of theano variable this list will contains, can be another list.
    depth
        Optionnal parameters, any value above 0 will create a nested list of
        this depth. (0-based)

    """

    def __init__(self, ttype, depth=0):

        if depth < 0:
            raise ValueError('Please specify a depth superior or'
                             'equal to 0')
        if not isinstance(ttype, gof.Type):
            raise TypeError('Expected a Theano Type')

        if depth == 0:
            self.ttype = ttype
        else:
            self.ttype = TypedListType(ttype, depth - 1)

    def filter(self, x, strict=False, allow_downcast=None):
        """

        Parameters
        ----------
        x
            Value to filter.
        strict
            If true, only native python list will be accepted.
        allow_downcast
            Does not have any utility at the moment.

        """
        if strict:
            if not isinstance(x, list):
                raise TypeError('Expected a python list')
        else:
            x = [self.ttype.filter(y) for y in x]

            if all(self.ttype.is_valid_value(y) for y in x):
                return x

            else:
                raise TypeError('Expected all elements to'
                                ' be %s' % str(self.ttype))

    def __eq__(self, other):
        """
        Two lists are equal if they contain the same type.

        """
        return type(self) == type(other) and self.ttype == other.ttype

    def __hash__(self):
        return gof.hashtype(self) ^ hash(self.ttype)

    def __str__(self):
        return 'TypedList <' + str(self.ttype) + '>'

    def get_depth(self):
        """
        Utilitary function to get the 0 based level of the list.

        """
        if isinstance(self.ttype, TypedListType):
            return self.ttype.get_depth() + 1
        else:
            return 0

    def values_eq(self, a, b):
        if not len(a) == len(b):
            return False

        for x in range(len(a)):
            if not self.ttype.values_eq(a[x], b[x]):
                return False

        return True

    def may_share_memory(self, a, b):
        if a is b:
            return True
        # As a list contain other element, if a or b isn't a list, we
        # still need to check if that element is contained in the
        # other list.
        if not isinstance(a, list):
            a = [a]
        if not isinstance(b, list):
            b = [b]
        for idx1 in range(len(a)):
            for idx2 in range(len(b)):
                if self.ttype.may_share_memory(a[idx1], b[idx2]):
                    return True

    def c_declare(self, name, sub, check_input=True):
        return """
        PyListObject* %(name)s;
        """ % dict(name=name)

    def c_init(self, name, sub):
        return """
        %(name)s = NULL;
        """ % dict(name=name)

    def c_extract(self, name, sub, check_input=True):
        if check_input:
            pre = """
            if (!PyList_Check(py_%(name)s)) {
                PyErr_SetString(PyExc_TypeError, "expected a list");
                %(fail)s
            }""" % dict(name=name, fail=sub['fail'])
        else:
            pre = ""
        return pre + """
        %(name)s = (PyListObject*) (py_%(name)s);
        """ % dict(name=name, fail=sub['fail'])

    def c_sync(self, name, sub):

        return """
        Py_XDECREF(py_%(name)s);
        py_%(name)s = (PyObject*)(%(name)s);
        Py_INCREF(py_%(name)s);
        """ % dict(name=name)

    def c_cleanup(self, name, sub):
        return ""

    def c_code_cache_version(self):
        return (2,)

    dtype = property(lambda self: self.ttype)
    ndim = property(lambda self: self.ttype.ndim + 1)
