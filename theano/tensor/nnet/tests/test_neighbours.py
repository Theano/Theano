from __future__ import absolute_import, print_function, division
import numpy
import unittest

import theano
from theano import shared, function
import theano.tensor as T
from theano.tensor.nnet.neighbours import images2neibs, neibs2images, Images2Neibs

from theano.tests import unittest_tools

mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpu')


class T_Images2Neibs(unittest_tools.InferShapeTester):
    mode = mode_without_gpu
    op = Images2Neibs
    dtypes = ['int64', 'float32', 'float64']

    def test_neibs(self):
        for shape, pshape in [((10, 7, 18, 18), (2, 2)),
                              ((10, 7, 6, 18), (3, 2)),
                              ((5, 7, 66, 66), (33, 33)),
                              ((5, 7, 68, 66), (34, 33))]:
            for border in ['valid', 'ignore_borders']:
                for dtype in self.dtypes:
                    images = shared(
                        numpy.arange(numpy.prod(shape), dtype=dtype).reshape(shape))
                    neib_shape = T.as_tensor_variable(pshape)

                    f = function([],
                                 images2neibs(images, neib_shape, mode=border),
                                 mode=self.mode)

                    # print images.get_value(borrow=True)
                    neibs = f()
                    # print neibs
                    g = function([],
                                 neibs2images(neibs, neib_shape, images.shape),
                                 mode=self.mode)
                    assert any([isinstance(node.op, self.op)
                                for node in f.maker.fgraph.toposort()])

                    # print g()
                    assert numpy.allclose(images.get_value(borrow=True), g())

    def test_neibs_manual(self):
        shape = (2, 3, 4, 4)
        for dtype in self.dtypes:
            images = shared(
                numpy.arange(numpy.prod(shape), dtype=dtype).reshape(shape))
            neib_shape = T.as_tensor_variable((2, 2))

            for border in ['valid', 'ignore_borders']:
                f = function([], images2neibs(images, neib_shape, mode=border),
                             mode=self.mode)
                assert any([isinstance(node.op, self.op)
                            for node in f.maker.fgraph.toposort()])

                # print images.get_value(borrow=True)
                neibs = f()
                # print neibs
                assert numpy.allclose(neibs, [
                    [0, 1, 4, 5],
                    [2, 3, 6, 7],
                    [8, 9, 12, 13],
                    [10, 11, 14, 15],
                    [16, 17, 20, 21],
                    [18, 19, 22, 23],
                    [24, 25, 28, 29],
                    [26, 27, 30, 31],
                    [32, 33, 36, 37],
                    [34, 35, 38, 39],
                    [40, 41, 44, 45],
                    [42, 43, 46, 47],
                    [48, 49, 52, 53],
                    [50, 51, 54, 55],
                    [56, 57, 60, 61],
                    [58, 59, 62, 63],
                    [64, 65, 68, 69],
                    [66, 67, 70, 71],
                    [72, 73, 76, 77],
                    [74, 75, 78, 79],
                    [80, 81, 84, 85],
                    [82, 83, 86, 87],
                    [88, 89, 92, 93],
                    [90, 91, 94, 95]])
                g = function([], neibs2images(neibs, neib_shape, images.shape),
                             mode=self.mode)

                assert numpy.allclose(images.get_value(borrow=True), g())

    def test_neibs_manual_step(self):
        shape = (2, 3, 5, 5)
        for dtype in self.dtypes:
            images = shared(numpy.asarray(numpy.arange(numpy.prod(
                            shape)).reshape(shape), dtype=dtype))
            neib_shape = T.as_tensor_variable((3, 3))
            neib_step = T.as_tensor_variable((2, 2))
            for border in ['valid', 'ignore_borders']:
                f = function([],
                             images2neibs(images, neib_shape, neib_step,
                                          mode=border),
                             mode=self.mode)

                neibs = f()
                assert self.op in [type(node.op)
                                   for node in f.maker.fgraph.toposort()]

                assert numpy.allclose(neibs, [
                    [0, 1, 2, 5, 6, 7, 10, 11, 12],
                    [2, 3, 4, 7, 8, 9, 12, 13, 14],
                    [10, 11, 12, 15, 16, 17, 20, 21, 22],
                    [12, 13, 14, 17, 18, 19, 22, 23, 24],
                    [25, 26, 27, 30, 31, 32, 35, 36, 37],
                    [27, 28, 29, 32, 33, 34, 37, 38, 39],
                    [35, 36, 37, 40, 41, 42, 45, 46, 47],
                    [37, 38, 39, 42, 43, 44, 47, 48, 49],
                    [50, 51, 52, 55, 56, 57, 60, 61, 62],
                    [52, 53, 54, 57, 58, 59, 62, 63, 64],
                    [60, 61, 62, 65, 66, 67, 70, 71, 72],
                    [62, 63, 64, 67, 68, 69, 72, 73, 74],
                    [75, 76, 77, 80, 81, 82, 85, 86, 87],
                    [77, 78, 79, 82, 83, 84, 87, 88, 89],
                    [85, 86, 87, 90, 91, 92, 95, 96, 97],
                    [87, 88, 89, 92, 93, 94, 97, 98, 99],
                    [100, 101, 102, 105, 106, 107, 110, 111, 112],
                    [102, 103, 104, 107, 108, 109, 112, 113, 114],
                    [110, 111, 112, 115, 116, 117, 120, 121, 122],
                    [112, 113, 114, 117, 118, 119, 122, 123, 124],
                    [125, 126, 127, 130, 131, 132, 135, 136, 137],
                    [127, 128, 129, 132, 133, 134, 137, 138, 139],
                    [135, 136, 137, 140, 141, 142, 145, 146, 147],
                    [137, 138, 139, 142, 143, 144, 147, 148, 149]])

                # neibs2images do not seam to support step != neib_shape
                # g = function([], neibs2images(neibs, neib_shape, images.shape),
                #             mode=self.mode)

                # print g()
                # assert numpy.allclose(images.get_value(borrow=True), g())

    def test_neibs_bad_shape(self):
        shape = (2, 3, 10, 10)
        for dtype in self.dtypes:
            images = shared(numpy.arange(
                numpy.prod(shape), dtype=dtype).reshape(shape))

            for neib_shape in [(3, 2), (2, 3)]:
                neib_shape = T.as_tensor_variable(neib_shape)
                f = function([], images2neibs(images, neib_shape),
                             mode=self.mode)
                self.assertRaises(TypeError, f)

                # Test that ignore border work in that case.
                f = function([],
                             images2neibs(images, neib_shape,
                                          mode='ignore_borders'),
                             mode=self.mode)
                assert self.op in [type(node.op)
                                   for node in f.maker.fgraph.toposort()]
                f()

    def test_neibs_wrap_centered_step_manual(self):

        expected1 = [[24, 20, 21, 4, 0, 1, 9, 5, 6],
                     [21, 22, 23, 1, 2, 3, 6, 7, 8],
                     [23, 24, 20, 3, 4, 0, 8, 9, 5],
                     [9, 5, 6, 14, 10, 11, 19, 15, 16],
                     [6, 7, 8, 11, 12, 13, 16, 17, 18],
                     [8, 9, 5, 13, 14, 10, 18, 19, 15],
                     [19, 15, 16, 24, 20, 21, 4, 0, 1],
                     [16, 17, 18, 21, 22, 23, 1, 2, 3],
                     [18, 19, 15, 23, 24, 20, 3, 4, 0]]
        expected2 = [[24, 20, 21, 4, 0, 1, 9, 5, 6],
                     [22, 23, 24, 2, 3, 4, 7, 8, 9],
                     [14, 10, 11, 19, 15, 16, 24, 20, 21],
                     [12, 13, 14, 17, 18, 19, 22, 23, 24]]
        expected3 = [[19, 15, 16, 24, 20, 21, 4, 0, 1, 9, 5, 6, 14, 10, 11],
                     [17, 18, 19, 22, 23, 24, 2, 3, 4, 7, 8, 9, 12, 13, 14],
                     [9, 5, 6, 14, 10, 11, 19, 15, 16, 24, 20, 21, 4, 0, 1],
                     [7, 8, 9, 12, 13, 14, 17, 18, 19, 22, 23, 24, 2, 3, 4]]
        expected4 = [[23, 24, 20, 21, 22, 3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
                     [21, 22, 23, 24, 20, 1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
                     [13, 14, 10, 11, 12, 18, 19, 15, 16, 17, 23, 24, 20, 21, 22],
                     [11, 12, 13, 14, 10, 16, 17, 18, 19, 15, 21, 22, 23, 24, 20]]
        expected5 = [[24, 20, 21, 4, 0, 1, 9, 5, 6],
                     [22, 23, 24, 2, 3, 4, 7, 8, 9],
                     [9, 5, 6, 14, 10, 11, 19, 15, 16],
                     [7, 8, 9, 12, 13, 14, 17, 18, 19],
                     [19, 15, 16, 24, 20, 21, 4, 0, 1],
                     [17, 18, 19, 22, 23, 24, 2, 3, 4]]
        expected6 = [[24, 20, 21, 4, 0, 1, 9, 5, 6],
                     [21, 22, 23, 1, 2, 3, 6, 7, 8],
                     [23, 24, 20, 3, 4, 0, 8, 9, 5],
                     [14, 10, 11, 19, 15, 16, 24, 20, 21],
                     [11, 12, 13, 16, 17, 18, 21, 22, 23],
                     [13, 14, 10, 18, 19, 15, 23, 24, 20]]

        # TODO test discontinous image

        for shp_idx, (shape, neib_shape, neib_step, expected) in enumerate([
            [(7, 8, 5, 5), (3, 3), (2, 2), expected1],
            [(7, 8, 5, 5), (3, 3), (3, 3), expected2],
            [(7, 8, 5, 5), (5, 3), (3, 3), expected3],
            [(7, 8, 5, 5), (3, 5), (3, 3), expected4],
            [(80, 90, 5, 5), (3, 3), (2, 3), expected5],
            [(1025, 9, 5, 5), (3, 3), (3, 2), expected6],
            [(1, 1, 5, 1035), (3, 3), (3, 3), None],
            [(1, 1, 1045, 5), (3, 3), (3, 3), None], ]
        ):

            for dtype in self.dtypes:

                images = shared(numpy.asarray(numpy.arange(numpy.prod(
                                shape)).reshape(shape), dtype=dtype))
                neib_shape = T.as_tensor_variable(neib_shape)
                neib_step = T.as_tensor_variable(neib_step)
                expected = numpy.asarray(expected)

                f = function([], images2neibs(images, neib_shape, neib_step,
                                              mode="wrap_centered"),
                             mode=self.mode)
                neibs = f()

                if expected.size > 1:
                    for i in range(shape[0] * shape[1]):
                        assert numpy.allclose(
                            neibs[i * expected.shape[0]:(i + 1) * expected.shape[0], :],
                            expected + 25 * i), "wrap_centered"

                assert self.op in [type(node.op)
                                   for node in f.maker.fgraph.toposort()]

                # g = function([], neibs2images(neibs, neib_shape, images.shape), mode=self.mode)
                # TODO: why this is commented?
                # assert numpy.allclose(images.get_value(borrow=True), g())

    def test_neibs_bad_shape_wrap_centered(self):
        shape = (2, 3, 10, 10)

        for dtype in self.dtypes:
            images = shared(numpy.arange(
                numpy.prod(shape), dtype=dtype
                ).reshape(shape))

            for neib_shape in [(3, 2), (2, 3)]:
                neib_shape = T.as_tensor_variable(neib_shape)

                f = function([], images2neibs(images, neib_shape,
                                              mode="wrap_centered"),
                             mode=self.mode)
                self.assertRaises(TypeError, f)

            for shape in [(2, 3, 2, 3), (2, 3, 3, 2)]:
                images = shared(numpy.arange(numpy.prod(shape)).reshape(shape))
                neib_shape = T.as_tensor_variable((3, 3))
                f = function([], images2neibs(images, neib_shape,
                                              mode="wrap_centered"),
                             mode=self.mode)
                self.assertRaises(TypeError, f)

            # Test a valid shapes
            shape = (2, 3, 3, 3)
            images = shared(numpy.arange(numpy.prod(shape)).reshape(shape))
            neib_shape = T.as_tensor_variable((3, 3))

            f = function([],
                         images2neibs(images, neib_shape, mode="wrap_centered"),
                         mode=self.mode)
            f()

    def test_grad_wrap_centered(self):
        # It is not implemented for now. So test that we raise an error.
        shape = (2, 3, 6, 6)
        images_val = numpy.random.rand(*shape).astype('float32')

        def fn(images):
            return images2neibs(images, (3, 3), mode='wrap_centered')

        self.assertRaises(TypeError, unittest_tools.verify_grad,
                          fn, [images_val], mode=self.mode)

    def test_grad_valid(self):
        shape = (2, 3, 6, 6)
        images_val = numpy.random.rand(*shape).astype('float32')

        def fn(images):
            return images2neibs(images, (2, 2))

        unittest_tools.verify_grad(fn, [images_val], mode=self.mode,
                                   eps=0.1)

        def fn(images):
            return images2neibs(images, (3, 2), (1, 2))

        unittest_tools.verify_grad(fn, [images_val], mode=self.mode,
                                   eps=0.1)

        def fn(images):
            return images2neibs(images, (1, 2), (5, 2))

        unittest_tools.verify_grad(fn, [images_val], mode=self.mode,
                                   eps=0.1)

    def test_grad_ignore_border(self):
        shape = (2, 3, 5, 5)
        images_val = numpy.random.rand(*shape).astype('float32')

        def fn(images):
            return images2neibs(images, (2, 2),
                                mode='ignore_borders')

        unittest_tools.verify_grad(fn, [images_val], mode=self.mode,
                                   eps=0.1)

    def test_neibs2images_grad(self):
        # say we had images of size (2, 3, 10, 10)
        # then we extracted 2x2 neighbors on this, we get (2 * 3 * 5 * 5, 4)
        neibs_val = numpy.random.rand(150, 4)

        def fn(neibs):
            return neibs2images(neibs, (2, 2), (2, 3, 10, 10))
        unittest_tools.verify_grad(fn, [neibs_val], mode=self.mode,
                                   eps=0.1)

    def test_neibs_valid_with_inconsistent_borders(self):
        shape = (2, 3, 5, 5)
        images = T.dtensor4()
        images_val = numpy.arange(numpy.prod(shape),
                                  dtype='float32').reshape(shape)

        def fn(images):
            return T.sum(T.sqr(images2neibs(images, (2, 2), mode='valid')),
                         axis=[0, 1])

        f = theano.function([images],
                            T.sqr(images2neibs(images, (2, 2), mode='valid')),
                            mode=self.mode)
        self.assertRaises(TypeError, f, images_val)

    def test_can_not_infer_nb_dim(self):
        # Was reported in gh-5613. Test that we do not crash
        # or that we crash in a few other case found while
        # investigating that case

        img = T.tensor4('img')
        patches = T.nnet.neighbours.images2neibs(img, [16, 16])
        extractPatches = theano.function([img], patches)

        patsRecovery = T.matrix('patsRecovery')
        original_size = T.ivector('original_size')

        for mode in ['valid', 'ignore_borders']:
            out = neibs2images(patsRecovery, (16, 16),
                               original_size, mode=mode)
            f = theano.function([patsRecovery, original_size], out)

            im_val = numpy.ones((1, 3, 320, 320), dtype=numpy.float32)
            neibs = extractPatches(im_val)
            f(neibs, im_val.shape)
            # Wrong number of dimensions
            self.assertRaises(ValueError, f, neibs,
                              (1, 1, 3, 320, 320))
            # End up with a step of 0
            self.assertRaises(ValueError, f, neibs,
                              (3, 320, 320, 1))

    def speed_neibs(self):
        shape = (100, 40, 18, 18)
        images = shared(numpy.arange(numpy.prod(shape),
                                     dtype='float32').reshape(shape))
        neib_shape = T.as_tensor_variable((3, 3))

        f = function([], images2neibs(images, neib_shape),
                     mode=self.mode)

        for i in range(1000):
            f()

    def speed_neibs_wrap_centered(self):
        shape = (100, 40, 18, 18)
        images = shared(numpy.arange(numpy.prod(shape),
                                     dtype='float32').reshape(shape))
        neib_shape = T.as_tensor_variable((3, 3))

        f = function([],
                     images2neibs(images, neib_shape, mode="wrap_centered"),
                     mode=self.mode)

        for i in range(1000):
            f()

    def test_infer_shape(self):
        shape = (100, 40, 6, 3)
        images = numpy.ones(shape).astype('float32')
        x = T.ftensor4()
        self._compile_and_check(
            [x], [images2neibs(x, neib_shape=(2, 1), mode='valid')],
            [images], Images2Neibs)
        self._compile_and_check(
            [x], [images2neibs(x, neib_shape=(2, 3), mode='valid')],
            [images], Images2Neibs)
        shape = (100, 40, 5, 4)
        images = numpy.ones(shape).astype('float32')
        x = T.ftensor4()
        self._compile_and_check(
            [x], [images2neibs(
                x, neib_shape=(2, 1), mode='ignore_borders')],
            [images], Images2Neibs)
        shape = (100, 40, 5, 3)
        images = numpy.ones(shape).astype('float32')
        x = T.ftensor4()
        self._compile_and_check(
            [x], [images2neibs(
                x, neib_shape=(2, 3), mode='ignore_borders')],
            [images], Images2Neibs)

        shape = (100, 40, 6, 7)
        images = numpy.ones(shape).astype('float32')
        x = T.ftensor4()
        self._compile_and_check(
            [x], [images2neibs(
                x, neib_shape=(2, 2), mode='ignore_borders')],
            [images], Images2Neibs)
        shape = (100, 40, 5, 10)
        images = numpy.ones(shape).astype('float32')
        x = T.ftensor4()
        self._compile_and_check(
            [x], [images2neibs(
                x, neib_shape=(3, 3), mode='wrap_centered')],
            [images], Images2Neibs)

if __name__ == '__main__':
    unittest.main()
