
import unittest
from core import build_mode, pop_mode

from ops import *


class _testCase_add_build_mode(unittest.TestCase):
    def setUp(self):
        build_mode()
        numpy.random.seed(44)
    def tearDown(self):
        pop_mode()


class _testCase_dot(unittest.TestCase):
    def setUp(self):
        build_eval_mode()
        numpy.random.seed(44)
    def tearDown(self):
        pop_mode()

    @staticmethod
    def rand(*args):
        return numpy.random.rand(*args)

    def cmp_dot(self,x,y):
        if 0:
            def spec(x):
                x = numpy.asarray(x)
                return type(x), x.dtype, x.shape
            zspec = dot.specs(spec(x), spec(y))
            nz = numpy.dot(x,y)
            self.failUnless(zspec == spec(nz))
        self.failUnless(_approx_eq(dot(x,y), numpy.dot(x,y)))

    def cmp_dot_comp(self, x,y):
        x = numpy.asarray(x)
        y = numpy.asarray(y)
        z = dot(x,y)
        p = compile.single(z)
        if len(x.shape):
            x[:] = numpy.random.rand(*x.shape)
        else:
            x.fill(numpy.random.rand(*x.shape))
        if len(y.shape):
            y[:] = numpy.random.rand(*y.shape)
        else:
            y.fill(numpy.random.rand(*y.shape))
        p() # recalculate z
        self.failUnless(_approx_eq(z, numpy.dot(x,y)))

    def test_dot_0d_0d(self): self.cmp_dot(1.1, 2.2)
    def test_dot_0d_1d(self): self.cmp_dot(1.1, self.rand(5))
    def test_dot_0d_2d(self): self.cmp_dot(3.0, self.rand(6,7))
    def test_dot_0d_3d(self): self.cmp_dot(3.0, self.rand(8,6,7))
    def test_dot_1d_0d(self): self.cmp_dot(self.rand(5), 1.1 )
    def test_dot_1d_1d(self): self.cmp_dot(self.rand(5), self.rand(5))
    def test_dot_1d_2d(self): self.cmp_dot(self.rand(6), self.rand(6,7))
    def test_dot_1d_3d(self): self.cmp_dot(self.rand(6), self.rand(8,6,7))
    def test_dot_2d_0d(self): self.cmp_dot(self.rand(5,6), 1.0)
    def test_dot_2d_1d(self): self.cmp_dot(self.rand(5,6), self.rand(6))
    def test_dot_2d_2d(self): self.cmp_dot(self.rand(5,6), self.rand(6,7))
    def test_dot_2d_3d(self): self.cmp_dot(self.rand(5,6), self.rand(8,6,7))
    def test_dot_3d_0d(self): self.cmp_dot(self.rand(4,5,6), 1.0)
    def test_dot_3d_1d(self): self.cmp_dot(self.rand(4,5,6), self.rand(6))
    def test_dot_3d_2d(self): self.cmp_dot(self.rand(4,5,6), self.rand(6,7))
    def test_dot_3d_3d(self): self.cmp_dot(self.rand(4,5,6), self.rand(8,6,7))
    def test_dot_0d_0d_(self): self.cmp_dot_comp(1.1, 2.2)
    def test_dot_0d_1d_(self): self.cmp_dot_comp(1.1, self.rand(5))
    def test_dot_0d_2d_(self): self.cmp_dot_comp(3.0, self.rand(6,7))
    def test_dot_0d_3d_(self): self.cmp_dot_comp(3.0, self.rand(8,6,7))
    def test_dot_1d_0d_(self): self.cmp_dot_comp(self.rand(5), 1.1 )
    def test_dot_1d_1d_(self): self.cmp_dot_comp(self.rand(5), self.rand(5))
    def test_dot_1d_2d_(self): self.cmp_dot_comp(self.rand(6), self.rand(6,7))
    def test_dot_1d_3d_(self): self.cmp_dot_comp(self.rand(6), self.rand(8,6,7))
    def test_dot_2d_0d_(self): self.cmp_dot_comp(self.rand(5,6), 1.0)
    def test_dot_2d_1d_(self): self.cmp_dot_comp(self.rand(5,6), self.rand(6))
    def test_dot_2d_2d_(self): self.cmp_dot_comp(self.rand(5,6), self.rand(6,7))
    def test_dot_2d_3d_(self): self.cmp_dot_comp(self.rand(5,6), self.rand(8,6,7))
    def test_dot_3d_0d_(self): self.cmp_dot_comp(self.rand(4,5,6), 1.0)
    def test_dot_3d_1d_(self): self.cmp_dot_comp(self.rand(4,5,6), self.rand(6))
    def test_dot_3d_2d_(self): self.cmp_dot_comp(self.rand(4,5,6), self.rand(6,7))
    def test_dot_3d_3d_(self): self.cmp_dot_comp(self.rand(4,5,6), self.rand(8,6,7))

    def test_dot_fail_1_1(self):
        x = numpy.random.rand(5)
        y = numpy.random.rand(6)
        try:
            z = dot(x,y)
        except ValueError, e:
            self.failUnless(str(e) == 'objects are not aligned', e)
            return
        self.fail()

    def test_dot_fail_1_2(self):
        x = numpy.random.rand(5)
        y = numpy.random.rand(6,4)
        try:
            z = dot(x,y)
        except ValueError, e:
            self.failUnless(str(e) == 'objects are not aligned', e)
            return
        self.fail()
    def test_dot_fail_1_3(self):
        x = numpy.random.rand(5)
        y = numpy.random.rand(6,4,7)
        try:
            z = dot(x,y)
        except ValueError, e:
            self.failUnless(str(e) == 'objects are not aligned', e)
            return
        self.fail()
    def test_dot_fail_2_1(self):
        x = numpy.random.rand(5,4)
        y = numpy.random.rand(6)
        try:
            z = dot(x,y)
        except ValueError, e:
            self.failUnless(str(e) == 'objects are not aligned', e)
            return
        self.fail()
    def test_dot_fail_2_2(self):
        x = numpy.random.rand(5,4)
        y = numpy.random.rand(6,7)
        try:
            z = dot(x,y)
        except ValueError, e:
            self.failUnless(str(e) == 'objects are not aligned', e)
            return
        self.fail()
    def test_dot_fail_2_3(self):
        x = numpy.random.rand(5,4)
        y = numpy.random.rand(6,7,8)
        try:
            z = dot(x,y)
        except ValueError, e:
            self.failUnless(str(e) == 'objects are not aligned', e)
            return
        self.fail()
    def test_dot_fail_3_1(self):
        x = numpy.random.rand(5,4,3)
        y = numpy.random.rand(6)
        try:
            z = dot(x,y)
        except ValueError, e:
            self.failUnless(str(e) == 'objects are not aligned', e)
            return
        self.fail()
    def test_dot_fail_3_2(self):
        x = numpy.random.rand(5,4,3)
        y = numpy.random.rand(6,7)
        try:
            z = dot(x,y)
        except ValueError, e:
            self.failUnless(str(e) == 'objects are not aligned', e)
            return
        self.fail()
    def test_dot_fail_3_3(self):
        x = numpy.random.rand(5,4,3)
        y = numpy.random.rand(6,7,8)
        try:
            z = dot(x,y)
        except ValueError, e:
            self.failUnless(str(e) == 'objects are not aligned', e)
            return
        self.fail()

class gemm(omega_op):
    def destroy_map(self):
        return {self.out:[self.inputs[0]]}
    def impl(z, a, x, y, b):
        if b == 0.0:
            if a == 1.0:
                z[:] = numpy.dot(x,y)
            elif a == -1.0:
                z[:] = -numpy.dot(x,y)
            else:
                z[:] = a * numpy.dot(x,y)
        elif b == 1.0:
            if a == 1.0:
                z += numpy.dot(x,y)
            elif a == -1.0:
                z -= numpy.dot(x,y)
            else:
                z += a * numpy.dot(x,y)
        else:
            z *= b
            z += a * numpy.dot(x,y)
        return z[:]
    def grad(z, a, x, y, b, gz):
        raise NotImplemented
    def refresh(self, alloc = False):
        z,a,x,y,b = self.inputs
        self.out.shape = z.shape
        self.out.dtype = z.dtype
        if alloc:
            self.out.data = z.data
    def c_support_code(self):
        return blas.cblas_header_text()
    def c_libs(self):
        return blas.ldflags()
    def c_impl((_zin, _a, _x, _y, _b), (_z,)):
        check_ab = """
        {
        if ((_a->descr->type_num != PyArray_DOUBLE)
            && (_a->descr->type_num != PyArray_FLOAT))
            goto _dot_execute_fallback;

        if ((_b->descr->type_num != PyArray_DOUBLE)
            && (_b->descr->type_num != PyArray_FLOAT))
            goto _dot_execute_fallback;
        }
        """
        return blas.gemm_code( check_ab,
                '(_a->descr->type_num == PyArray_FLOAT) ? (REAL)(((float*)_a->data)[0]) : (REAL)(((double*)_a->data)[0])',
                '(_b->descr->type_num == PyArray_FLOAT) ? (REAL)(((float*)_b->data)[0]) : (REAL)(((double*)_b->data)[0])')


class _testCase_transpose(unittest.TestCase):

    def setUp(self):
        build_eval_mode()

    def tearDown(self):
        pop_mode()
    
    def test_1d_alias(self):
        a = numpy.ones(10)
        ta = transpose(a)
        self.failUnless(ta.data.shape == a.shape)
        self.failUnless(numpy.all(ta.data == a))
        a[3] *= -1.0
        self.failUnless(numpy.all(ta.data == a))

    def test_1d_copy(self):
        a = numpy.ones(10)
        ta = transpose_copy(a)
        self.failUnless(ta.data.shape == a.shape)
        self.failUnless(numpy.all(ta.data == a))
        a[3] *= -1.0
        self.failIf(numpy.all(ta.data == a))

    def test_2d_alias(self):
        a = numpy.ones((10,3))
        ta = transpose(a)
        self.failUnless(ta.data.shape == (3,10))

    def test_3d_alias(self):
        a = numpy.ones((10,3,5))
        ta = transpose(a)
        self.failUnless(ta.data.shape == (5,3,10))
        a[9,0,0] = 5.0
        self.failUnless(ta.data[0,0,9] == 5.0)

    def test_3d_copy(self):
        a = numpy.ones((10,3,5))
        ta = transpose_copy(a)
        self.failUnless(ta.data.shape == (5,3,10))
        a[9,0,0] = 5.0
        self.failUnless(ta.data[0,0,9] == 1.0)



class _testCase_power(unittest.TestCase):
    def setUp(self):
        build_eval_mode()
        numpy.random.seed(44)
    def tearDown(self):
        pop_mode()
    def test1(self):
        r = numpy.random.rand(50)
        exp_r = exp(r)
        self.failUnless(exp_r.__array__().__class__ is numpy.ndarray)

    def test_0(self):
        r = numpy.random.rand(50)

        exp_r = exp(r)
        n_exp_r = numpy.exp(r)
        self.failUnless( _approx_eq(exp_r, n_exp_r), 
                (exp_r, exp_r.data, n_exp_r,
                    numpy.max(numpy.abs(n_exp_r.__sub__(exp_r.__array__())))))

        log_exp_r = log(exp_r)
        self.failUnless( _approx_eq(log_exp_r, r), log_exp_r)

    def test_1(self):
        r = numpy.random.rand(50)
        r2 = pow(r,2)
        self.failUnless( _approx_eq(r2, r*r))



class _testCase_slicing(unittest.TestCase):
    def setUp(self):
        build_eval_mode()
    def tearDown(self):
        pop_mode()

    def test_getitem0(self):
        a = numpy.ones((4,4))
        wa1 = wrap(a)[:,1]
        try:
            err = wa1 + a
        except ValueError, e:
            self.failUnless(str(e) == \
                    'The dimensions of the inputs do not match.',
                    'Wrong ValueError')
            return
        self.fail('add should not have succeeded')

    def test_getitem1(self):
        a = numpy.ones((4,4))
        wa1 = wrap(a)[1]
        self.failUnless(wa1.data.shape == (4,))

    def test_getslice_0d_all(self):
        """Test getslice does not work on 0d array """
        a = numpy.ones(())
        try:
            wa1 = wrap(a)[:]
        except IndexError, e:
            self.failUnless(str(e) == "0-d arrays can't be indexed.")
            return
        self.fail()
    def test_getslice_1d_all(self):
        """Test getslice on 1d array"""
        a = numpy.ones(4)
        wa1 = wrap(a)[:]
        self.failUnless(wa1.data.shape == (4,), 'wrong shape')
        self.failUnless(numpy.all(wa1.data == a), 'unequal value')

        a[1] = 3.4
        self.failUnless(wa1.data[1] == 3.4, 'not a view')

        try:
            wa1[2] = 2.5
        except TypeError, e:
            self.failUnless("object does not support item assignment" in str(e))
            return
        self.fail()
    def test_getslice_3d_all(self):
        """Test getslice on 3d array"""
        a = numpy.ones((4,5,6))
        wa1 = wrap(a)[:]
        self.failUnless(wa1.data.shape == (4,5,6), 'wrong shape')
        self.failUnless(numpy.all(wa1.data == a), 'unequal value')

        a[1,1,1] = 3.4
        self.failUnless(wa1.data[1,1,1] == 3.4, 'not a view')
    def test_getslice_1d_some(self):
        """Test getslice on 1d array"""
        a = numpy.ones(5)
        wa1 = wrap(a)[1:3]
        a[2] = 5.0
        a[3] = 2.5
        self.failUnless(wa1.data.shape == (2,))
        self.failUnless(a[1] == wa1.data[0])
        self.failUnless(a[2] == wa1.data[1])
    def test_getslice_1d_step(self):
        """Test getslice on 1d array"""
        a = numpy.ones(8)
        wa1 = wrap(a)[0:8:2]
        for i in xrange(8): a[i] = i

        self.failUnless(wa1.shape == (4,))
        for i in xrange(4):
            self.failUnless(a[i*2] == wa1.data[i])
    def test_getslice_3d_float(self):
        """Test getslice on 3d array"""
        a = numpy.asarray(range(4*5*6))
        a.resize((4,5,6))
        wa1 = wrap(a)[1:3]
        self.failUnless(wa1.shape == (2,5,6))
        self.failUnless(numpy.all(a[1:3] == wa1.data))
        a[1] *= -1.0
        self.failUnless(numpy.all(a[1:3] == wa1.data))
    def test_getslice_3d_one(self):
        """Test getslice on 3d array"""
        a = numpy.asarray(range(4*5*6))
        a.resize((4,5,6))
        wa = wrap(a)
        wa_123 = wa[1,2,3]
        self.failUnless(wa_123.shape == (), wa_123.shape)



if __name__ == '__main__':
    unittest.main()

