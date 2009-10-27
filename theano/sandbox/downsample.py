""" Ops for downsampling images.

Planned: 
DownsampleFactorMax, DownsampleAvg, DownsampleSoftmax.

"""
#This file should move along with conv.py

from theano import sparse, gof, Op, tensor, Variable, Apply
from theano.printing import Print
import numpy
import __builtin__

class DownsampleFactorMaxGrad(Op):
    def __init__(self, ds, ignore_border):
        self.ds = tuple(ds)
        self.ignore_border = ignore_border

    def __eq__(self, other):
        return type(self) == type(other) and self.ds == other.ds and self.ignore_border == other.ignore_border

    def __hash__(self):
        return hash(type(self)) ^ hash(self.ds) ^ hash(self.ignore_border)

    def __str__(self):
        return '%s{%s,%s}' % (self.__class__.__name__, self.ds, self.ignore_border)

    def make_node(self, x, maxout, gz):
        # make_node should only be called by the grad function of DownsampleFactorMax, 
        # so these asserts should not fail.
        assert isinstance(x, Variable) and x.ndim==4
        assert isinstance(maxout, Variable) and maxout.ndim==4
        assert isinstance(gz, Variable) and gz.ndim==4

        return Apply(self, [x, maxout, gz], [x.type()])

    def perform(self, node, (x, maxout, gz), (gx_stg,)):
        gx = numpy.zeros_like(x)

        ds0, ds1 = self.ds
        shape2 = (x.shape[2] / ds0 * ds0) if self.ignore_border else x.shape[2]
        shape3 = (x.shape[3] / ds1 * ds1) if self.ignore_border else x.shape[3]
        for n in xrange(x.shape[0]):
            for k in xrange(x.shape[1]):
                for i in xrange(shape2):
                    zi = i / ds0
                    for j in xrange(shape3):
                        zj = j / ds1
                        gx[n,k,i,j] = gz[n,k,zi,zj] if (maxout[n,k,zi,zj] == x[n,k,i,j]) else 0
        gx_stg[0] = gx

    def c_code(self, node, name, (x, z, gz), (gx,), sub):
        fail = sub['fail']
        ignore_border = int(self.ignore_border)
        ds0, ds1 = self.ds
        return """
        int x_typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
        int z_typenum = PyArray_ObjectType((PyObject*)%(z)s, 0);
        int gz_typenum = PyArray_ObjectType((PyObject*)%(gz)s, 0);
        int x_shp0_usable;
        int x_shp1_usable;
        int z_shp0, z_shp1;
        if ((x_typenum != z_typenum) || (x_typenum != gz_typenum))
        {
            PyErr_SetString(PyExc_ValueError, "input types must all match");
            %(fail)s;
        }
        if(%(x)s->nd!=4) 
        {
            PyErr_SetString(PyExc_ValueError, "x must be a 4d ndarray");
            %(fail)s;
        }
        if(%(z)s->nd!=4) 
        {
            PyErr_SetString(PyExc_ValueError, "z must be a 4d ndarray");
            %(fail)s;
        }
        if(%(gz)s->nd!=4) 
        {
            PyErr_SetString(PyExc_ValueError, "gz must be a 4d ndarray");
            %(fail)s;
        }
        z_shp0 = %(z)s->dimensions[2];
        z_shp1 = %(z)s->dimensions[3];
        if (%(ignore_border)s)
        {
            x_shp0_usable = z_shp0 * %(ds0)s;
            x_shp1_usable = z_shp1 * %(ds1)s;
        }
        else
        {
            x_shp0_usable = %(x)s->dimensions[2];
            x_shp1_usable = %(x)s->dimensions[3];
        }
        if ((!%(gx)s)
          || *PyArray_DIMS(%(gx)s)!=4
          ||(%(gx)s->dimensions[0] != %(x)s->dimensions[0])
          ||(%(gx)s->dimensions[1] != %(x)s->dimensions[1])
          ||(%(gx)s->dimensions[2] != %(x)s->dimensions[2])
          ||(%(gx)s->dimensions[3] != %(x)s->dimensions[3])
          )
        {
          Py_XDECREF(%(gx)s);
          %(gx)s = (PyArrayObject*) PyArray_ZEROS(4, %(x)s->dimensions, x_typenum,0);
        }

        for(int b=0;b<%(x)s->dimensions[0];b++){
          for(int k=0;k<%(x)s->dimensions[1];k++){
            int mini_i = 0;
            int zi = 0;
            for(int i=0;i< x_shp0_usable; i++){
               int mini_j = 0;
               int zj = 0;
               for(int j=0; j< x_shp1_usable; j++){
                 dtype_%(x)s * __restrict__ xp = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,b,k,i,j)));
                 dtype_%(gx)s * __restrict__ gxp = ((dtype_%(gx)s*)(PyArray_GETPTR4(%(gx)s,b,k,i,j)));
                 dtype_%(z)s * __restrict__ zp = ((dtype_%(z)s*)(PyArray_GETPTR4(%(z)s,b,k,zi,zj)));
                 dtype_%(gz)s * __restrict__ gzp = ((dtype_%(gz)s*)(PyArray_GETPTR4(%(gz)s,b,k,zi,zj)));
                 gxp[0] = (zp[0] == xp[0]) ? gzp[0] : 0;
                 mini_j = (mini_j + 1 == %(ds1)s) ? 0 : mini_j+1;
                 zj += (mini_j == 0);
              }//for j
              mini_i = (mini_i + 1 == %(ds0)s) ? 0 : mini_i+1;
              zi += (mini_i == 0);

              for (int j = x_shp1_usable; j < %(x)s->dimensions[3]; ++j) {
                dtype_%(gx)s * gxp = ((dtype_%(gx)s*)(PyArray_GETPTR4(%(gx)s,b,k,i,j)));
                gxp[0] = 0;
              }
            }//for i

            for(int i = x_shp0_usable; i < %(x)s->dimensions[2]; i++){
                for (int j = 0; j < %(x)s->dimensions[3]; ++j) {
                    dtype_%(gx)s * gxp = ((dtype_%(gx)s*)(PyArray_GETPTR4(%(gx)s,b,k,i,j)));
                    gxp[0] = 0;
                }
            }
          }//for k
        }//for b
        """ %locals()

    def c_code_cache_version(self):
        return ()

                
class DownsampleFactorMax(Op):
    """
    For N-dimensional tensors, consider that the last two dimensions span images.
    This Op downsamples these images by taking the max over non-overlapping rectangular regions.
    """

    @staticmethod
    def out_shape(imgshape, ds, ignore_border=False):
        #old code not tested (not evenread)
        a, b = imgshape[-2:]
        rval = list(imgshape[:-2])+[ a/ds[0], b/ds[1]]
        if not ignore_border:
            if a % ds[0]:
                rval[-2] += 1
            if b % ds[1]:
                rval[-1] += 1
        return rval

    def __init__(self, ds, ignore_border=False):
        self.ds = tuple(ds)
        self.ignore_border = ignore_border

    def __eq__(self, other):
        return type(self) == type(other) and self.ds == other.ds and self.ignore_border == other.ignore_border

    def __hash__(self):
        return hash(type(self)) ^ hash(self.ds) ^ hash(self.ignore_border)

    def __str__(self):
        return '%s{%s,%s}' % (self.__class__.__name__, self.ds, self.ignore_border)

    def make_node(self, x):
        if x.type.ndim != 4:
            raise TypeError()
        # TODO: consider restrucing the dtype?
        return gof.Apply(self, [x], [x.type()])

    def perform(self, node, (x,), (z,)):
        """
        """
        if len(x.shape)!=4:
            raise NotImplementedError('DownsampleFactorMax requires 4D input for now')
        if z[0] is None:
            z[0] = numpy.zeros(self.out_shape(x.shape, self.ds, self.ignore_border)) -float('inf')
            z[0] = numpy.asarray(z[0], dtype=x.dtype)
        zz=z[0]
        ds0, ds1 = self.ds

        x_usable2 = (x.shape[2] / ds0 * ds0) if self.ignore_border else x.shape[2]
        x_usable3 = (x.shape[3] / ds1 * ds1) if self.ignore_border else x.shape[3]
        for n in xrange(x.shape[0]):
            for k in xrange(x.shape[1]):
                for i in xrange(x_usable2):
                    zi = i / ds0
                    for j in xrange(x_usable3):
                        zj = j / ds1
                        zz[n,k,zi,zj] = __builtin__.max(zz[n,k,zi,zj], x[n,k,i,j])

    def grad(self,(x,), (gz,)):
        maxout = self(x)
        return [DownsampleFactorMaxGrad(self.ds, ignore_border=self.ignore_border)(x, maxout, gz)]

    def c_code(self, node, name, (x,), (z, ), sub):
        fail=sub['fail']
        ignore_border = int(self.ignore_border)
        ds0, ds1 = self.ds
        return """
        int typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
        int x_shp0_usable;
        int x_shp1_usable;
        int z_shp0, z_shp1;
        if(%(x)s->nd!=4) 
        {
            PyErr_SetString(PyExc_ValueError, "x must be a 4d ndarray");
            %(fail)s;
        }
        z_shp0 = %(x)s->dimensions[2] / %(ds0)s;
        z_shp1 = %(x)s->dimensions[3] / %(ds1)s;
        if (%(ignore_border)s)
        {
            x_shp0_usable = z_shp0 * %(ds0)s;
            x_shp1_usable = z_shp1 * %(ds1)s;
        }
        else
        {
            z_shp0 += (%(x)s->dimensions[2] %% %(ds0)s) ? 1 : 0;
            z_shp1 += (%(x)s->dimensions[3] %% %(ds1)s) ? 1 : 0;
            x_shp0_usable = %(x)s->dimensions[2];
            x_shp1_usable = %(x)s->dimensions[3];
        }
        if ((!%(z)s)
          || *PyArray_DIMS(%(z)s)!=4
          ||(%(z)s->dimensions[0] != %(x)s->dimensions[0])
          ||(%(z)s->dimensions[1] != %(x)s->dimensions[1])
          ||(%(z)s->dimensions[2] != z_shp0)
          ||(%(z)s->dimensions[3] != z_shp1)
          )
        {
          if (%(z)s) Py_XDECREF(%(z)s);
          npy_intp dims[4] = {0,0,0,0};
          dims[0]=%(x)s->dimensions[0];
          dims[1]=%(x)s->dimensions[1];
          dims[2]=z_shp0;
          dims[3]=z_shp1;
          %(z)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum,0); //TODO: zeros not necessary
        }

        if (z_shp0 && z_shp1)
        {
            for(int b=0;b<%(x)s->dimensions[0];b++){
              for(int k=0;k<%(x)s->dimensions[1];k++){
                int mini_i = 0;
                int zi = 0;
                for(int i=0;i< x_shp0_usable; i++){
                  int mini_j = 0;
                  int zj = 0;
                  for(int j=0; j<x_shp1_usable; j++){
                    dtype_%(x)s a = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,b,k,i,j)))[0];
                    dtype_%(z)s * __restrict__ z = ((dtype_%(z)s*)(PyArray_GETPTR4(%(z)s,b,k,zi,zj)));
                    z[0] = (((mini_j|mini_i) == 0) || z[0] < a) ? a : z[0];
                    mini_j = ((mini_j + 1) == %(ds1)s) ? 0 : mini_j+1;
                    zj += (mini_j == 0);
                  }
                  mini_i = ((mini_i + 1) == %(ds0)s) ? 0 : mini_i+1;
                  zi += (mini_i == 0);
                }
              }
            }
        }
        """ % locals()

    def c_code_cache_version(self):
        return ()

def max_pool(images=None, imshp=None, maxpoolshp=None, ignore_border=True):
    """Implements a max pooling layer

    Uses the same API as sp.max_pool but uses the Downsample op instead.

    Takes as input a 2D tensor of shape batch_size x img_size and performs max pooling.
    Max pooling downsamples by taking the max value in a given area, here defined by
    maxpoolshp. Outputs a 2D tensor of shape batch_size x output_size.

    Parameters are keyword arguments in order to use func_to_mod.

    @param images: 2D tensor containing images on which to apply convolution.
                   Assumed to be of shape batch_size x img_size
    @param imgshp: tuple containing image dimensions
    @param maxpoolshp: tuple containing shape of area to max pool over
    
    @output out1: symbolic result (2D tensor)
    @output out2: logical shape of the output

    """
    if len(imshp) == 2:
        imshp = (1,) + imshp
    elif len(imshp)!=3:
        raise NotImplementedError("!")
    
    # all these reshapes should happen in place
    imrshp = tensor.stack(images.shape[0],
                          *[tensor.as_tensor(x) for x in imshp])
    imtensor = tensor.reshape(images, imrshp)

    maxpop = DownsampleFactorMax(maxpoolshp, ignore_border)
    rval = maxpop(imtensor)

    return tensor.flatten(rval,2), maxpop.out_shape(imshp, maxpoolshp, ignore_border)
