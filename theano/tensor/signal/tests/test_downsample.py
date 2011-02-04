import unittest, sys, time
import numpy
import theano.tensor as tensor
from theano.tests import unittest_tools as utt
from theano.tensor.signal.downsample import DownsampleFactorMax, max_pool_2d
from theano import function, Mode


class TestDownsampleFactorMax(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    @staticmethod
    def numpy_max_pool_2d(input, ds, ignore_border=False):
        '''Helper function, implementing max_pool_2d in pure numpy'''
        if len(input.shape) < 2:
            raise NotImplementedError('input should have at least 2 dim, shape is %s'\
                    % str(input.shape))

        xi=0
        yi=0
        if not ignore_border:
            if input.shape[-2] % ds[0]:
                xi += 1
            if input.shape[-1] % ds[1]:
                yi += 1

        out_shp = list(input.shape[:-2])
        out_shp.append(input.shape[-2]/ds[0]+xi)
        out_shp.append(input.shape[-1]/ds[1]+yi)

        output_val = numpy.zeros(out_shp)

        for k in numpy.ndindex(input.shape[:-2]):
            for i in range(output_val.shape[-2]):
                ii =  i*ds[0]
                for j in range(output_val.shape[-1]):
                    jj = j*ds[1]
                    patch = input[k][ii:ii+ds[0],jj:jj+ds[1]]
                    output_val[k][i,j] = numpy.max(patch)
        return output_val

    def test_DownsampleFactorMax(self):
        rng = numpy.random.RandomState(utt.fetch_seed())

        # generate random images
        maxpoolshps = ((1,1),(2,2),(3,3),(2,3))
        imval = rng.rand(4,10,64,64)
        images = tensor.dtensor4()

        for maxpoolshp in maxpoolshps:
            for ignore_border in [True,False]:
                print 'maxpoolshp =', maxpoolshp
                print 'ignore_border =', ignore_border

                ## Pure Numpy computation
                numpy_output_val = self.numpy_max_pool_2d(imval, maxpoolshp, ignore_border)

                output = max_pool_2d(images, maxpoolshp, ignore_border)
                f = function([images,],[output,])
                output_val = f(imval)
                assert numpy.all(output_val == numpy_output_val)

                #DownsampleFactorMax op
                maxpool_op = DownsampleFactorMax(maxpoolshp, ignore_border=ignore_border)(images)
                f = function([images], maxpool_op)
                output_val = f(imval)
                assert (numpy.abs(output_val - numpy_output_val) < 1e-5).all()

    def test_DownsampleFactorMax_grad(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        maxpoolshps = ((1,1),(3,2),(2,3))
        imval = rng.rand(2,3,3,4) * 10.0 #more variance means numeric gradient will be more accurate

        for maxpoolshp in maxpoolshps:
            for ignore_border in [True,False]:
                print 'maxpoolshp =', maxpoolshp
                print 'ignore_border =', ignore_border
                def mp(input):
                    return DownsampleFactorMax(maxpoolshp, ignore_border=ignore_border)(input)
                utt.verify_grad(mp, [imval], rng=rng)

    def test_max_pool_2d_2D(self):
        rng = numpy.random.RandomState(utt.fetch_seed())

        maxpoolshps = ((1,1),(3,2))
        imval = rng.rand(4,5)
        images = tensor.dmatrix()

        for maxpoolshp in maxpoolshps:
            for ignore_border in [True,False]:
                print 'maxpoolshp =', maxpoolshp
                print 'ignore_border =', ignore_border
                numpy_output_val = self.numpy_max_pool_2d(imval, maxpoolshp, ignore_border)

                output = max_pool_2d(images, maxpoolshp, ignore_border)
                output_val = function([images], output)(imval)
                assert numpy.all(output_val == numpy_output_val)

                def mp(input):
                    return max_pool_2d(input, maxpoolshp, ignore_border)
                utt.verify_grad(mp, [imval], rng=rng)

    def test_max_pool_2d_3D(self):
        rng = numpy.random.RandomState(utt.fetch_seed())

        maxpoolshps = [(1,2)]
        imval = rng.rand(2,3,4)
        images = tensor.dtensor3()

        for maxpoolshp in maxpoolshps:
            for ignore_border in [True,False]:
                print 'maxpoolshp =', maxpoolshp
                print 'ignore_border =', ignore_border
                numpy_output_val = self.numpy_max_pool_2d(imval, maxpoolshp, ignore_border)

                output = max_pool_2d(images, maxpoolshp, ignore_border)
                output_val = function([images], output)(imval)
                assert numpy.all(output_val == numpy_output_val)

                c = tensor.sum(output)
                c_val = function([images], c)(imval)

                g = tensor.grad(c, images)
                g_val = function([images],
                        [g.shape,
                            tensor.min(g, axis=(0,1,2)),
                            tensor.max(g, axis=(0,1,2))]
                        )(imval)

#removed as already tested in test_max_pool_2d_2D
#This make test in debug mode too slow.
#                def mp(input):
#                    return max_pool_2d(input, maxpoolshp, ignore_border)
#                utt.verify_grad(mp, [imval], rng=rng)


    def test_max_pool_2d_6D(self):
        rng = numpy.random.RandomState(utt.fetch_seed())

        maxpoolshps = [(3,2)]
        imval = rng.rand(2,1,1,1,3,4)
        images = tensor.TensorType('float64', [False]*6)()

        for maxpoolshp in maxpoolshps:
            for ignore_border in [True,False]:
                print 'maxpoolshp =', maxpoolshp
                print 'ignore_border =', ignore_border
                numpy_output_val = self.numpy_max_pool_2d(imval, maxpoolshp, ignore_border)

                output = max_pool_2d(images, maxpoolshp, ignore_border)
                output_val = function([images], output)(imval)
                assert numpy.all(output_val == numpy_output_val)

#removed as already tested in test_max_pool_2d_2D
#This make test in debug mode too slow.
#                def mp(input):
#                    return max_pool_2d(input, maxpoolshp, ignore_border)
#                utt.verify_grad(mp, [imval], rng=rng)



if __name__ == '__main__':
    unittest.main()
