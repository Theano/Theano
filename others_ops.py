## This file contain ops that are not currently integrated in the core of threano. 
## Not all of those ops have been throughtfully tested.

from theano.scalar import *
import theano.tensor as T
import theano
from numpy import *
class Prepend_scalar_constant_to_each_row(Op):
    def __init__(self, val = 0):
        if isinstance(val, float):
            val = constant(val)
        self.val = val

    def make_node(self, mat):
        #check type of input
        if not isinstance(mat,Result) or not mat.type==T.matrix().type:
            raise TypeError("Expected a matrix as input")
        x = T.as_tensor(mat)
        y = T.as_tensor(self.val)
        if x.type.dtype != y.type.dtype:
            TypeError("the value to prepend don't have the same type as the matrix")
        
        node = Apply(op=self, inputs=[mat], outputs=[T.matrix()])
        return node

    def perform(self, node, (mat, ), (output, )):
        if output[0] == None:
            output[0]=numpy.empty((mat.shape[0],mat.shape[1]+1),dtype=mat.dtype)
            out=output[0]
        else:
            out=output[0]
            assert out.shape==(mat.shape[0],mat.shape[1]+1)

        out[:,0].fill(self.val.data)
        out[:,1:]=mat

    def grad(self, (mat,), (goutput,)):
        return goutput[:,1:]

class Prepend_scalar_to_each_row(Op):        
    def make_node(self, val, mat):
        #check type of input
        if isinstance(val, float):
            val = constant(val)
        if not isinstance(mat,Result) or not mat.type==T.matrix().type:
            raise TypeError("Expected a matrix as input")
        x = T.as_tensor(mat)
        y = T.as_tensor(val)
        if x.type.dtype != y.type.dtype:
            TypeError("the value to prepend don't have the same type as the matrix")
        
        node = Apply(op=self, inputs=[val,mat], outputs=[T.matrix()])
        return node

    def perform(self, node, (val,mat), (output, )):
        if output[0] == None:
            output[0]=numpy.empty((mat.shape[0],mat.shape[1]+1),dtype=mat.dtype)
            out=output[0]
        else:
            out=output[0]
            assert out.shape==(mat.shape[0],mat.shape[1]+1)
        out[:,0].fill(val)
        out[:,1:]=mat

    def grad(self, (val, mat), (goutput,)):
        return goutput[:,0], goutput[:,1:]

if __name__ == '__main__':

    x=T.matrix('x')
    y=Prepend_scalar_constant_to_each_row(4.)(x)
    f=theano.function([x],[y])
    mat=numpy.random.rand(3,5)
    gradient(f(mat))
    print f(mat)

    x=T.matrix('x')
    y=Prepend_scalar_to_each_row()(5.,x)
    f=theano.function([x],[y])
    mat=numpy.ones((3,5),dtype="float32")
    print f(mat)

