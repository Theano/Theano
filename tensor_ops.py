
from gof import Op, utils, Destroyer, Viewer
import gof.op

from tensor import *



###########################
#### Binary Operations ####
###########################

#########
## Dot ##
#########






class Dot(TensorOp):
    @staticmethod
    def _output_shape(xshape, yshape):
        # This describes the logic to calculate numpy.dot(x, y).shape
        # given x.shape and y.shape
        if len(xshape) == 0: # x is a scalar
            shape = yshape
        else:
            if len(yshape) >= 2: #y is a matrix or tensor
                assert xshape[-1] == yshape[-2]
                shape = tuple(xshape[:-1]+ yshape[:-2]+yshape[-1:])
            elif len(yshape)==1: #y is vector
                assert xshape[-1] == yshape[-1]
                shape = tuple(xshape[:-1])
            else:                #y is a scalar
                shape = xshape
        return shape

    def impl(self, x, y):
        return numpy.dot(x, y)
    def grad(self, (x, y), gz):
        return dot(gz, transpose(y)), dot(transpose(x), gz)
    def propagate_broadcastable(self, x, y):
        assert len(x) == 2 and len(x) == len(y)
        return [(x[0], y[1])]
    def c_support_code(self):
        return blas.cblas_header_text()
    def c_libs(self):
        return blas.ldflags()
    def c_impl(self, (_x, _y), (_z, )):
        return blas.gemm_code('', '1.0', '0.0')






############
## Others ##
############



