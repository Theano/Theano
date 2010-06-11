import theano
from theano import Op, Apply
import theano.tensor as T
from theano.tensor.opt import register_specialize
from theano.gof import local_optimizer

from theano.sandbox.cuda import cuda_available
if cuda_available:
    from theano.sandbox.cuda import CudaNdarrayType
    from theano.sandbox.cuda.basic_ops import host_from_gpu, gpu_from_host

class Images2Neibs(Op):
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
    def make_node(self, ten4, neib_shape):
        ten4 = T.as_tensor_variable(ten4)
        neib_shape = T.as_tensor_variable(neib_shape)
        return Apply(self, [ten4, neib_shape], [T.matrix()])

    def grad(self, (pvals, unis), (gz,)):
        return [None, None]

    def c_code_cache_version(self):
        return ()
        #return (1,)
                
    def c_code(self, node, name, (ten4, neib_shape), (z,), sub):

        fail = sub['fail']
        return """
        {
        if (%(ten4)s->nd != 4)
        {
            PyErr_Format(PyExc_TypeError, "ten4 wrong rank");
            %(fail)s;
        }
        if (%(neib_shape)s->nd != 1)
        {
            PyErr_Format(PyExc_TypeError, "neib_shape wrong rank");
            %(fail)s;
        }
        if ( (%(neib_shape)s->dimensions)[0] != 2)
        {
            PyErr_Format(PyExc_TypeError, "neib_shape wrong shape ; has to contain 2 elements");
            %(fail)s;
        }
        
        const npy_intp c = (npy_intp) *(dtype_%(neib_shape)s*) PyArray_GETPTR1(%(neib_shape)s, 0);
        const npy_intp d = (npy_intp) *(dtype_%(neib_shape)s*) PyArray_GETPTR1(%(neib_shape)s, 1);

        if ( (%(ten4)s->dimensions)[2] %% c != 0)
        {
            PyErr_Format(PyExc_TypeError, "neib_shape[0] must divide ten4.shape[2]");
            %(fail)s;
        }
        if ( (%(ten4)s->dimensions)[3] %% d != 0)
        {
            PyErr_Format(PyExc_TypeError, "neib_shape[1] must divide ten4.shape[3]");
            %(fail)s;
        }
        
        // new dimensions for z
        const npy_intp z_dim1 = c * d;
        const npy_intp z_dim0 =  (%(ten4)s->dimensions)[2] / c
                            * (%(ten4)s->dimensions)[3] / d
                            * (%(ten4)s->dimensions)[1]
                            * (%(ten4)s->dimensions)[0];
        
        if ((NULL == %(z)s)
            || ((%(z)s->dimensions)[0] != z_dim0 )
            || ((%(z)s->dimensions)[1] != z_dim1 )
        )
        {
            Py_XDECREF(%(z)s);
            npy_intp dims[2];
            dims[0] = z_dim0;
            dims[1] = z_dim1;
            
            %(z)s = (PyArrayObject*) PyArray_EMPTY(2,
                dims,
                type_num_%(ten4)s,
                0);
                       
            if (!%(z)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc z output");
                %(fail)s;
            }
        }
        }

        { // NESTED SCOPE
        
        const int nb_batch = (%(ten4)s->dimensions)[0];
        const int nb_stack = (%(ten4)s->dimensions)[1];
        const int height = (%(ten4)s->dimensions)[2];
        const int width = (%(ten4)s->dimensions)[3];
        
        // (c,d) = neib_shape
        const int c = (int) *(dtype_%(neib_shape)s*) PyArray_GETPTR1(%(neib_shape)s, 0);
        const int d = (int) *(dtype_%(neib_shape)s*) PyArray_GETPTR1(%(neib_shape)s, 1);

        const int grid_c = height/c;
        const int grid_d = width/d;

        // Oh this is messed up...      
        for (int n = 0; n < nb_batch; n++)              // loop over batches
            for (int s = 0; s < nb_stack; s++)          // loop over stacks
                for (int a = 0; a < grid_c; a++)        // loop over height/c
                    for (int b = 0; b < grid_d; b++)    // loop over width/d
                        for (int i = 0; i < c; i++)     // loop over c
                           for (int j = 0; j < d; j++)  // loop over d
                            {
                                int ten4_2 = i + a * c;
                                int ten4_3 = j + b * d;
                                int z_row = b + grid_d*(a + grid_c*(s + nb_stack*n));
                                int z_col = j + d * i;
                                //printf("\\n(%%i,%%i,%%i,%%i) --> (%%i,%%i)",n,s, ten4_2, ten4_3, z_row, z_col);
                                dtype_%(z)s* curr_z = (dtype_%(z)s*) PyArray_GETPTR2(%(z)s, z_row, z_col);
                                *curr_z = *( (dtype_%(ten4)s*) PyArray_GETPTR4(%(ten4)s, n, s, ten4_2, ten4_3));
                                //printf("%%f ", *curr_z);
                            }
        } // END NESTED SCOPE
        """ % locals()
images2neibs = Images2Neibs()

def neibs2images(neibs, neib_shape, original_shape):
    """
    Inverse of images2neib.
    
    neibs : matrix like the one obtained by images2neib
    neib_shape : neib_shape that was used in images2neib
    original_shape : original shape of the 4d tensor given to images2neib
    
    Return a 4d tensor of shape `original_shape`.
    """
    neibs = T.as_tensor_variable(neibs)
    neib_shape = T.as_tensor_variable(neib_shape)
    original_shape = T.as_tensor_variable(original_shape)
    
    new_neib_shape = T.stack( original_shape[-1]/neib_shape[1], neib_shape[1] )
    return images2neibs(neibs.dimshuffle('x','x',0,1), new_neib_shape).reshape(original_shape)