from theano import tensor, function
import numpy


if __name__ == "__main__":

    x = tensor.dtensor3()
    y = tensor.ivector()
    
    a = numpy.random.rand(4, 6, 3)
    
    for shp in [(2, 3, 3, 4), (1, 2, 3, 2, 3, 2), (1, 3, 4, 6),
                (3, 8, 3, 1), (3, 24), (24, 3), (72,)]:

        f_order = function([x, y],
                    tensor.reshape(x, y, ndim=len(shp), order='F'))

        c_order = function([x, y],
                    tensor.reshape(x, y, ndim=len(shp), order='C'))

        f_out = f_order(a, shp)
        c_out = c_order(a, shp)

        assert numpy.allclose(f_out, numpy.reshape(a, shp, order='F'))
        assert numpy.allclose(c_out, numpy.reshape(a, shp, order='C'))


    x = tensor.dvector()
    y = tensor.ivector()
    
    a = numpy.random.rand(72)
    
    for shp in [(2, 3, 3, 4), (1, 2, 3, 2, 3, 2), (1, 3, 4, 6),
                (3, 8, 3, 1), (3, 24), (24, 3), (72,)]:

        f_order = function([x, y],
                    tensor.reshape(x, y, ndim=len(shp), order='F'))

        c_order = function([x, y],
                    tensor.reshape(x, y, ndim=len(shp), order='C'))

        f_out = f_order(a, shp)
        c_out = c_order(a, shp)

        assert numpy.allclose(f_out, numpy.reshape(a, shp, order='F'))
        assert numpy.allclose(c_out, numpy.reshape(a, shp, order='C'))
