
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





if __name__ == '__main__':
    unittest.main()

