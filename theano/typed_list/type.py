from theano import gof


class TypedListType(gof.Type):

    def __init__(self, ttype, depth=0):
        """
        :Parameters:
            -'ttype' : Type of theano variable this list
            will contains, can be another list.
            -'depth' : Optionnal parameters, any value
            above 0 will create a nested list of this
            depth. (0-based)
        """
        if depth < 0:
            raise ValueError('Please specify a depth superior or'
                            'equal to 0')
        if not hasattr(ttype, 'is_valid_value'):
            raise TypeError('Expected a Theano type')

        if depth == 0:
            self.ttype = ttype
        else:
            self.ttype = TypedListType(ttype, depth - 1)

        self.Variable.ttype = self.ttype

    def filter(self, x, strict=False, allow_downcast=None):
        """
        :Parameters:
            -'x' : value to filter
            -'strict' : if true, only native python list will be accepted
            -'allow_downcast' : does not have any utility at the moment
        """
        if strict:
            if not isinstance(x, list):
                raise TypeError('Expected a python list')
        else:
            x = list(x)

            x = [self.ttype.filter(y) for y in x]

            if all(self.ttype.is_valid_value(y) for y in x):
                return x

            else:
                raise TypeError('Expected all elements to'
                                ' be %s' % str(self.ttype))

    def __eq__(self, other):
        """
        two list are equals if they contains the same type.
        """

        if not hasattr(other, 'ttype'):
            return False

        return  (self.ttype == other.ttype)

    def __str__(self):
        return 'Typed List <' + str(self.ttype) + '>'

    def get_depth(self):
        """
        utilitary function to get the 0 based
        level of the list
        """
        if hasattr(self.ttype, 'get_depth'):
            return self.ttype.get_depth() + 1
        else:
            return 0

    def make_variable(self, name=None):
        """Return a `TypedListVariable` of this type

        :Parameters:
         - `name`: str
           A pretty name to identify this `Variable` when printing and
           debugging
        """
        return self.Variable(self, name=name)
