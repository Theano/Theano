import numpy
import unittest

import theano
from theano import tensor

def makeSharedTester(shared_constructor_,
                     dtype_,
                     get_value_borrow_true_alias_,
                     shared_borrow_true_alias_,
                     set_value_borrow_true_alias_,
                     internal_type_,
                     test_internal_type_,
                     theano_fct_,
                     ref_fct_,
                     cast_value_ = numpy.asarray,
                     add_matrix_ = False):
    """
    This is a generic fct to allow reusing the same test function
    for many shared variable of many types.

    We must use /= as sparse type don't support other inplace operation.
    """
    class SharedTester(unittest.TestCase):
        shared_constructor = staticmethod(shared_constructor_)
        dtype = dtype_
        get_value_borrow_true_alias = get_value_borrow_true_alias_
        shared_borrow_true_alias = shared_borrow_true_alias_
        internal_type = internal_type_
        test_internal_type = staticmethod(test_internal_type_)
        theano_fct = staticmethod(theano_fct_)
        ref_fct = staticmethod(ref_fct_)
        set_value_borrow_true_alias = set_value_borrow_true_alias_
        cast_value = staticmethod(cast_value_)
        add_matrix = add_matrix_

        def test_shared_dont_alias(self):
            dtype = self.dtype
            if dtype is None:
                dtype = theano.config.floatX

            rng = numpy.random.RandomState([3,5,17])
            x = numpy.asarray(rng.uniform(0,1,[2,4]),dtype=dtype)
            x = self.cast_value(x)

            x_ref = self.ref_fct(x)
            x_shared = self.shared_constructor(x, borrow = False)
            total = self.theano_fct(x_shared)

            total_func = theano.function([],total)

            total_val = total_func()

            assert numpy.allclose(self.ref_fct(x), total_val)

            x /= .5

            total_val_2 = total_func()

            #value used to construct should not alias with internal
            assert numpy.allclose(total_val, total_val_2)

            x = x_shared.get_value(borrow = False)

            x /= .5

            total_val_3 = total_func()

            #value returned by access should not alias with internal
            assert numpy.allclose(total_val, total_val_3)

            #in this case we can alias
            x = x_shared.get_value(borrow = True)
            x /= .5

            #this is not required by the contract but it is a feature we've
            #implemented for some type of SharedVariable.
            if self.get_value_borrow_true_alias:
                assert numpy.allclose(self.ref_fct(x), total_func())
            else:
                assert numpy.allclose(x_ref, total_func())


        def test_return_internal_type(self):
            dtype = self.dtype
            if dtype is None:
                dtype = theano.config.floatX

            rng = numpy.random.RandomState([3,5,17])
            x = numpy.asarray(rng.uniform(0,1,[2,4]),dtype=dtype)
            x = self.cast_value(x)

            x_ref = self.ref_fct(x)
            x_shared = self.shared_constructor(x, borrow = False)
            total = self.theano_fct(x_shared)

            total_func = theano.function([],total)

            #in this case we can alias with the internal value
            x = x_shared.get_value(borrow = True, return_internal_type = True)
            assert self.test_internal_type(x)

            values_to_add = .5
            if self.add_matrix:
                values_to_add = self.internal_type(numpy.ones(x.shape,dtype=dtype)/2)#supported for cudandarray, but not ndarray.
            x /= values_to_add#supported by ndarray and CudaNdarray

            #this is not required by the contract but it is a feature we can
            #implement for some type of SharedVariable.
            assert numpy.allclose(self.ref_fct(x), total_func())

            x = x_shared.get_value(borrow = False, return_internal_type = True)
            assert self.test_internal_type(x)
            assert x is not x_shared.container.value
            x /= values_to_add#supported by ndarray and CudaNdarray

            #this is required by the contract
            assert not numpy.allclose(self.ref_fct(x), total_func())

        def test_set_value(self):
            dtype = self.dtype
            if dtype is None:
                dtype = theano.config.floatX

            rng = numpy.random.RandomState([3,5,17])
            x = numpy.asarray(rng.uniform(0,1,[2,4]),dtype=dtype)
            x = self.cast_value(x)

            x_orig = x
            x_orig_copy = x.copy()
            x_ref = self.ref_fct(x)
            x_shared = self.shared_constructor(x, borrow = False)
            total = self.theano_fct(x_shared)

            total_func = theano.function([],total)

            #test if that theano shared variable optimize set_value(borrow=True)
            get_x = x_shared.get_value(borrow=True)
            assert get_x is not x_orig#borrow=False to shared_constructor
            get_x /= .5
            x_shared.set_value(get_x, borrow=True)
            x = x_shared.get_value(borrow=True)
            if self.set_value_borrow_true_alias:
                assert x is get_x
            else:
                assert x is not get_x
            assert numpy.allclose(self.ref_fct(x_orig/.5),self.ref_fct(x))

            #test optimized get set value on the gpu(don't pass data to the cpu)
            get_x = x_shared.get_value(borrow=True, return_internal_type=True)
            assert get_x is not x_orig#borrow=False to shared_constructor
            assert self.test_internal_type(get_x)
            values_to_add = .5
            if self.add_matrix:
                values_to_add = self.internal_type(numpy.ones(x.shape,dtype=dtype)/2)#supported for cudandarray, but not ndarray.
                assert self.test_internal_type(values_to_add)

            get_x /= values_to_add#supported by ndarray and CudaNdarray
            assert self.test_internal_type(get_x)
            x_shared.set_value(get_x, borrow=True)
            x = x_shared.get_value(borrow=True, return_internal_type=True)
            assert self.test_internal_type(x)
            assert x is get_x

            ################ TODO test Out.
        def test_shared_do_alias(self):
            dtype = self.dtype
            if dtype is None:
                dtype = theano.config.floatX

            rng = numpy.random.RandomState([2,4,16])
            x = numpy.asarray(rng.uniform(1,2,[4,2]),dtype=dtype)
            x = self.cast_value(x)
            x_ref = self.ref_fct(x)

            x_shared = self.shared_constructor(x, borrow = True)

            total = self.theano_fct(x_shared)

            total_func = theano.function([],total)

            total_val = total_func()

            assert numpy.allclose(self.ref_fct(x), total_val)

            x /= .5

            #not required by the contract but it is a feature we've implemented
            if self.shared_borrow_true_alias:
                assert numpy.allclose(self.ref_fct(x), total_func())
            else:
                assert numpy.allclose(x_ref, total_func())

    return SharedTester

test_shared_options=makeSharedTester(tensor.shared, 'float64',
                                     True, True, True,
                                     numpy.ndarray,
                                     lambda a: isinstance(a,numpy.ndarray),
                                     theano.tensor.sum,
                                     numpy.sum)

