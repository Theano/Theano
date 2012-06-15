import numpy
from theano import tensor, function


class TestKeepDims:

    def makeKeepDims_local(self, x, y, axis):
        x = tensor.as_tensor_variable(x)
        y = tensor.as_tensor_variable(y)

        if axis is None:
            axis = numpy.arange(x.ndim)
        i = 0
        new_dims = []
        for j, _ in enumerate(x.shape):
            if j in axis:
                new_dims.append('x')
            else:
                new_dims.append(i)
                i += 1

        return tensor.DimShuffle(y.type.broadcastable, new_dims)(y)

    def test_keepdims(self):

        x = tensor.dtensor3()
        a = numpy.random.rand(3, 2, 4)

        # 'max_and_argmax' has two outputs and can be specified with either
        # a single or every axis:
        for axis in [[0], [1], [2]]:

            op = tensor.max_and_argmax
            keep_param = function([x], op(x, axis=axis, keepdims=True)[0])
            keep_synth = function([x], self.makeKeepDims_local(x,
                                op(x, axis=axis, keepdims=False)[0], axis))

            # FRED: choisir l'une ou l'autre de ces verifications:
            assert numpy.allclose(keep_param(a), keep_synth(a))
            assert keep_param(a).shape == keep_synth(a).shape

            keep_param = function([x], op(x, axis=axis, keepdims=True)[1])
            keep_synth = function([x], self.makeKeepDims_local(x,
                                op(x, axis=axis, keepdims=False)[1], axis))

            # FRED: choisir l'une ou l'autre de ces verifications:
            assert numpy.allclose(keep_param(a), keep_synth(a))
            assert keep_param(a).shape == keep_synth(a).shape

        # the following ops can be specified with either a single axis or every
        # axis:
        for op in ([tensor.argmax, tensor.max, tensor.argmin, tensor.min]):

            for axis in [[0], [1], [2]]:

                keep_param = function([x], op(x, axis=axis, keepdims=True))
                keep_synth = function([x], self.makeKeepDims_local(x,
                                op(x, axis=axis, keepdims=False), axis))

                # FRED: choisir l'une ou l'autre de ces verifications:
                assert numpy.allclose(keep_param(a), keep_synth(a))
                assert keep_param(a).shape == keep_synth(a).shape

            keep_param = function([x], op(x, axis=None, keepdims=True))
            keep_synth = function([x], self.makeKeepDims_local(x,
                                op(x, axis=None, keepdims=False), None))

            # FRED: choisir l'une ou l'autre de ces verifications:
            assert numpy.allclose(keep_param(a), keep_synth(a))
            assert keep_param(a).shape == keep_synth(a).shape

        # the following ops can be specified with a freely specified axis
        # parameter
        for op in ([tensor.sum, tensor.prod, tensor.mean, tensor.var,
                    tensor.std, tensor.all, tensor.any]):
        # FRED: il faudra  ajouter les ops suivantes a la boucle ci-dessus:
        # tensor.all, tensor.any
        # Celles-ci semblent presentement defectueuses puisqu'elles plantent
        # a la compilation dans un interpreteur distinct.

            for axis in [[0], [1], [2], [0, 1], [1, 2], [0, 1, 2]]:

                keep_param = function([x], op(x, axis=axis, keepdims=True))
                keep_synth = function([x], self.makeKeepDims_local(x,
                                op(x, axis=axis, keepdims=False), axis))

                # FRED: choisir l'une ou l'autre de ces verifications:
                assert numpy.allclose(keep_param(a), keep_synth(a))
                assert keep_param(a).shape == keep_synth(a).shape

            keep_param = function([x], op(x, axis=None, keepdims=True))
            keep_synth = function([x], self.makeKeepDims_local(x,
                                op(x, axis=None, keepdims=False), None))

            # FRED: choisir l'une ou l'autre de ces verifications:
            assert numpy.allclose(keep_param(a), keep_synth(a))
            assert keep_param(a).shape == keep_synth(a).shape


if __name__ == '__main__':

    TestKeepDims().test_keepdims()