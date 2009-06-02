import numpy as N
import theano
import theano.tensor as T
from theano import gof, Op, tensor
from theano.printing import Print

def getFilterOutShp(inshp, kshp, (dx,dy)=(1,1), mode='valid'):
    s = -1 if mode=='valid' else 1
    inshp, kshp = N.array(inshp), N.array(kshp)
    return  N.int64(N.ceil((inshp[1:] + s*kshp - s*1)/\
            N.array([dy,dx], dtype='float')))

class ConvOp(Op):
    """
    A convolution op that should mimic scipy.signal.convolve2d, but faster!
    In development.
    """

    def __init__(self, imshp, kshp, nkern, bsize, dx, dy, output_mode='valid', unroll_batch=0, unroll_kern=0):
        """
        unroll_batch. If >0 will use a version that will unroll the batch loop by the value of the option. By default don't use this version of the code.
        unroll_nkern. idem as unroll_batch but unroll the kernel loop.
        """
        imshp = tuple(imshp)
        if len(imshp)==2:
            self.imshp = (1,)+imshp
        elif len(imshp)==3:
            self.imshp = imshp
        else:
            raise Exception("bad len for imshp")

        self.kshp = tuple(kshp)
        self.nkern = nkern
        self.bsize=bsize
        self.dx=dx
        self.dy=dy

        self.unroll_batch=unroll_batch
        self.unroll_kern=unroll_kern

        if self.unroll_batch>0 and self.bsize % self.unroll_batch!=0:
            raise Exception("unroll_batch(%s) should be 0 or a multiple of bsize(%s)"%(str(self.unroll_batch),str(self.bsize)))
        if self.unroll_kern>0 and self.nkern % unroll_kern!=0:
            raise Exception("unroll_kern(%s) should be 0 or a multiple of nkern(%s)"%(str(self.unroll_kern),str(self.nkern)))
        if self.dx!=1 or self.dy!=1:
            print "Warning, dx!=1 or dy!=1 only supported in python mode!"
            raise NotImplementedError()
        self.outshp = getFilterOutShp(self.imshp, kshp, (dx,dy), output_mode)
        self.out_mode = output_mode
        if not self.out_mode in ["valid", "full"]:
            raise Exception("Mode %s not implemented"%self.out_mode)
       
        assert (self.outshp >= 0).all()

#    def __eq__(self, other):
#        raise Error("Not implemented")

#    def __hash__(self):
#        raise Error("Not implemented")

    def make_node(self, inputs, kerns):
        # TODO: find a way to make ConvOp work for N-D (after NIPS09)
        outdim = kerns.ndim
        output = tensor.tensor(dtype=inputs.type.dtype, broadcastable=[False]*outdim);

        return gof.Apply(self, [inputs, kerns], [output])

    def perform(self,node, (img2d, filtersflipped), (z,)):
        """
        By default if len(img2d.shape)==3, we
        """
        # TODO: move these back out to global scope when they no longer cause an atexit error
        from scipy.signal.signaltools import  _valfrommode, _bvalfromboundary
        from scipy.signal.sigtools import _convolve2d
        if z[0] is None:
            z[0] = N.zeros((self.bsize,)+(self.nkern,)+tuple(self.outshp))
        zz=z[0]
        val = _valfrommode(self.out_mode)
        bval = _bvalfromboundary('fill')

        img2d = img2d.reshape((self.bsize,)+ self.imshp)
        filtersflipped = filtersflipped.reshape((self.nkern,self.imshp[0])+self.kshp)
        
        for b in range(self.bsize):
            for n in range(self.nkern):
                zz[b,n,...].fill(0)
                for im0 in range(self.imshp[0]):
                    zz[b,n,...] +=  _convolve2d(\
                        img2d[b,im0,...], filtersflipped[n,im0,...],1,val, bval, 0)
        zz = zz[:,:,0::self.dx,0::self.dy]
        z[0]=zz


    def grad(self, (inputs, kerns), (gz,)):
        """
        In development. Works for test cases in test_sp.py
        A few known issues:
        * doesn't work for rectangular images or filters
        * inputs needs to be a 4D tensor. Couldn't get 3D to work
        * will crash if filter the same size as input image
        """

        ####### Determine gradient on kernels ########
        if inputs.ndim == 3:
            inputs = tensor.shape_padleft(inputs,1)

        newin = tensor.DimShuffle(inputs.broadcastable, (1,0,2,3))(inputs)
        newgz = tensor.DimShuffle(gz.broadcastable, (1,0,2,3))(gz)
    
        if self.out_mode == 'valid':
            (img, filters) = (newin, newgz)
            (bsize, nkern) = (self.imshp[0], self.nkern)
            imshp = N.hstack((self.bsize, self.imshp[1:]))
            kshp  = self.outshp[::-1]
        elif self.out_mode == 'full':
            (img, filters) = (newgz, newin)
            (bsize, nkern) = (self.nkern, self.imshp[0])
            imshp = N.hstack((self.bsize, self.outshp))
            kshp  = self.imshp[1:][::-1]
        else:
            raise NotImplementedError('Only [full,valid] modes are currently supported.')

        filters = filters[:,:,::-1,::-1]

        dw = ConvOp(imshp, kshp, nkern, bsize, 1,1, output_mode='valid')(img,filters)
        if self.out_mode == 'valid':
            # before DimShuffle, dw is of shape visdim x nkern x kshp[0] x kshp[1]
            dw = tensor.DimShuffle(dw.broadcastable, (1,0,2,3))(dw)
            dw = dw[:,:,::-1,::-1]

        ####### Determine gradient on inputs ########
        mode = 'valid' if self.out_mode == 'full' else 'full'
        filters = tensor.DimShuffle(gz.broadcastable, (1,0,2,3))(kerns)
        filters = filters[:,:,::-1,::-1]
        nkern = self.imshp[0]
        imshp = N.hstack((self.nkern,self.outshp))
        din = ConvOp(imshp, self.kshp, nkern, self.bsize, 
                     1,1, output_mode=mode)(gz,filters)

        return [din, dw]

#def c():
    def c_headers(self):
        return ['"Python.h"', '"numpy/noprefix.h"']

    def c_support_code(self):
        return """
#define STRIDES(arr) ((arr)->strides)
#define FULL  2
#define SAME  1
#define VALID 0
#include <iostream>
using namespace std;
""" + tensor.blas.blas_header_text()
    def c_libraries(self):
        return tensor.blas.ldflags()
    def c_code(self, node, name, (img2d, filtersflipped), (z, ), sub):
        if node.inputs[0].type.dtype != node.inputs[1].type.dtype:
            raise NotImplementedError()
        d=locals()
        d.update(sub)
        d["self_out_mode"]=self.out_mode
        d["self_bsize"]=self.bsize
        d["self_nkern"]=self.nkern
        d["self_dx"]=self.dx
        d["self_dy"]=self.dy
        d["self_outshp0"]=self.outshp[0]
        d["self_outshp1"]=self.outshp[1]
        d["self_imshp0"]=self.imshp[0]
        d["self_imshp1"]=self.imshp[1]
        d["self_imshp2"]=self.imshp[2]
        d["self_kshp0"]=self.kshp[0]
        d["self_kshp1"]=self.kshp[1]
        d["affectation"]="=" if self.imshp[0]==1 else "+="
        if node.inputs[0].type.dtype=="float32": d["type"]="float"
        elif node.inputs[0].type.dtype=="float64": d["type"]="double"
        else: raise Exception("Type %s not implemented"%node.inputs[0].type.dtype)
        if self.unroll_kern>0 and self.unroll_batch>0:
            print "return unrolled batch and kern code by",self.unroll_batch, self.unroll_kern
            return gen_conv_code_unroll_batch_kern(d, self.unroll_batch,
                                                   self.unroll_kern)
        elif self.unroll_kern>0:
            print "return unrolled kern code by",self.unroll_kern
            return gen_conv_code_unroll_kern(d, self.unroll_kern)
        elif self.unroll_batch>0:
            print "return unrolled batch code by",self.unroll_batch
            return gen_conv_code_unroll_batch(d, self.unroll_batch)

        #TODO: should we choose the unroll size automatically with the bigger divisor under 5? under 10?
        if self.out_mode == 'valid':
            return _conv_op_code_valid_gemm % d
        else:
            return _conv_op_code_a % d

def convolve2(kerns, kshp, nkern, images, imshp, bsize, step=(1,1),
              bias=None, mode='valid'):

    # if imshp, is a tuple, images contains one input dimension
    nvis_dim = 1 if len(imshp)!=3 else imshp[0]

    # all these reshapes should happen in place
    imrshp   = tensor.as_tensor([bsize] + list(imshp))
    imtensor = tensor.reshape(images, imrshp)

    kernrshp   = tensor.as_tensor([nkern, nvis_dim] + list(kshp))
    kerntensor = tensor.reshape(kerns, kernrshp)
 
    convop = ConvOp(imshp, kshp, nkern, bsize, 1, 1, output_mode=mode)
    convout = convop(imtensor, kerntensor)
   
    if bias:
        biastensor = tensor.DimShuffle((False,), ('x',0,'x','x'), inplace=True)(bias)
        convout = convout + biastensor
        
    rval = tensor.flatten(convout, 2)
    return rval, N.hstack((nkern, convop.outshp))


_conv_op_code_a = """
int mode=-1,typenum=0, typenum_f=0;
PyArrayObject *ain1=NULL, *ain2=NULL, *filtersflipped_arr=NULL, *img2d_arr=NULL;
const %(type)s fill_value = 0;

int type_im=PyArray_TYPE(%(img2d)s);
int type_ker=PyArray_TYPE(%(filtersflipped)s);

npy_intp dim_zz[2]={%(self_outshp0)s,%(self_outshp1)s};
npy_intp dim_im[2]={%(self_imshp1)s,%(self_imshp2)s};
npy_intp dim_ker[2]={%(self_kshp0)s,%(self_kshp1)s};

PyArray_Dims img2d_shape;
npy_intp img2d_dim[4]={1,1,0,0};
img2d_shape.ptr=img2d_dim;
img2d_shape.len=4;

PyArray_Dims kerns_shape;
npy_intp kerns_dim[4]={1,1,0,0};
kerns_shape.ptr=kerns_dim;
kerns_shape.len=4;
PyObject *img2d=NULL, *contig, *filtersflipped=NULL;
string s="%(self_out_mode)s";

if(%(img2d)s->nd==2){
  img2d_dim[3]=%(img2d)s->dimensions[1];
  img2d_dim[2]=%(img2d)s->dimensions[0];
}else if(%(img2d)s->nd==3){
  img2d_dim[3]=%(img2d)s->dimensions[2];
  img2d_dim[2]=%(img2d)s->dimensions[1];
  img2d_dim[0]=%(img2d)s->dimensions[0];
}else if(%(img2d)s->nd==4){
  img2d_dim[3]=%(img2d)s->dimensions[3];
  img2d_dim[2]=%(img2d)s->dimensions[2];
  img2d_dim[1]=%(img2d)s->dimensions[1];
  img2d_dim[0]=%(img2d)s->dimensions[0];
}else {
    PyErr_SetString(PyExc_ValueError, "img don't have a good shape");
    %(fail)s;
}

if(%(filtersflipped)s->nd==3){
  kerns_dim[3]=%(filtersflipped)s->dimensions[2];
  kerns_dim[2]=%(filtersflipped)s->dimensions[1];
  kerns_dim[0]=%(filtersflipped)s->dimensions[0];
}else if(%(filtersflipped)s->nd==4){
  kerns_dim[3]=%(filtersflipped)s->dimensions[3];
  kerns_dim[2]=%(filtersflipped)s->dimensions[2];
  kerns_dim[1]=%(filtersflipped)s->dimensions[1];
  kerns_dim[0]=%(filtersflipped)s->dimensions[0];
}else{
    PyErr_SetString(PyExc_ValueError, "kernel don't have a good shape");
    %(fail)s;
}

img2d = PyArray_Newshape(%(img2d)s,&img2d_shape, PyArray_CORDER);
img2d_arr = (PyArrayObject*)img2d;
if ((img2d_arr->strides[3] != sizeof(%(type)s)) 
     || (img2d_arr->strides[2] != img2d_arr->dimensions[3]*sizeof(%(type)s))){
    contig = (PyObject*)(PyArray_GETCONTIGUOUS((PyArrayObject*)img2d));
    Py_DECREF(img2d);
    img2d = contig;
    if (!PyArray_ISCONTIGUOUS(img2d)){
        PyErr_SetString(PyExc_ValueError, "img2d isn't contiguous");
        %(fail)s;
    }
}
img2d_arr = (PyArrayObject*)img2d;

filtersflipped = PyArray_Newshape(%(filtersflipped)s,&kerns_shape, PyArray_CORDER);
filtersflipped_arr = (PyArrayObject*)filtersflipped;
if ((filtersflipped_arr->strides[3] != sizeof(%(type)s)) 
     || (filtersflipped_arr->strides[2] != filtersflipped_arr->dimensions[3]*sizeof(%(type)s))){
    contig = (PyObject*)(PyArray_GETCONTIGUOUS((PyArrayObject*)filtersflipped));
    Py_DECREF(filtersflipped);
    filtersflipped = contig;
    if (!PyArray_ISCONTIGUOUS(filtersflipped)){
        PyErr_SetString(PyExc_ValueError, "filtersflipped isn't contiguous");
        %(fail)s;
    }
}
filtersflipped_arr = (PyArrayObject*)filtersflipped;

if(s=="valid") mode=0;
else if(s=="full") mode=2;
else {PyErr_SetString(PyExc_ValueError, "invalid mode, only full and valid are supported"); %(fail)s;};
typenum = PyArray_ObjectType((PyObject*)%(img2d)s, 0);
typenum_f = PyArray_ObjectType((PyObject*)%(filtersflipped)s, 0);
if (typenum < 0) {PyErr_SetString(PyExc_ValueError, "Invalid type"); %(fail)s;}
if (typenum != typenum_f) {PyErr_SetString(PyExc_ValueError, "Input types must match"); %(fail)s;}

if (!img2d) %(fail)s;
if (!filtersflipped) %(fail)s;
if ((!%(z)s)
  || *PyArray_DIMS(%(z)s)!=4
  ||(%(z)s->dimensions[0] != %(self_bsize)s)
  ||(%(z)s->dimensions[1] != %(self_nkern)s)
  ||(%(z)s->dimensions[2] != dim_zz[0])
  || (%(z)s->dimensions[3] != dim_zz[1])
  )
{
  if (%(z)s) Py_DECREF(%(z)s);
  npy_intp dims[4] = {0,0,0,0};
  if(!dims) %(fail)s;
  dims[0]=%(self_bsize)s;
  dims[1]=%(self_nkern)s;
  dims[2]=dim_zz[0];
  dims[3]=dim_zz[1];
  %(z)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum,0);
}else{
  //PyArray_FILLWBYTE((PyObject*)%(z)s,0);
}

int Os[2];
if (mode == FULL) {Os[0] = dim_im[0]+dim_ker[0]-1; Os[1] = dim_im[1]+dim_ker[1]-1;}
else {Os[0] = dim_im[0]-dim_ker[0]+1; Os[1] = dim_im[1]-dim_ker[1]+1;}

for(int b=0;b< %(self_bsize)s;b++){
  for(int n_kern=0;n_kern<%(self_nkern)s;n_kern++){

    //assertions
    if (%(z)s->strides[0] != %(z)s->dimensions[1] *%(z)s->dimensions[2] *%(z)s->dimensions[3] * sizeof(%(type)s)) %(fail)s;
    if (%(z)s->strides[1] != %(z)s->dimensions[2] * %(z)s->dimensions[3] * sizeof(%(type)s)) %(fail)s;
    if (%(z)s->strides[2] != %(z)s->dimensions[3] * sizeof(%(type)s)) %(fail)s;
    if (%(z)s->strides[3] != sizeof(%(type)s)) %(fail)s;

    %(type)s * __restrict__ out=(%(type)s *)(PyArray_GETPTR2(%(z)s,b,n_kern));
    for (int i = 0; i < dim_zz[0]*dim_zz[1]; ++i) out[i] = 0;

    for(int stack_size=0;stack_size<%(self_imshp0)s;stack_size++){

      const %(type)s * __restrict__ in=(%(type)s *)(PyArray_GETPTR2(img2d,b,stack_size));
      const %(type)s * __restrict__ hvals=(%(type)s *)(PyArray_GETPTR2(filtersflipped,n_kern,stack_size));

      int new_m;

      for (int m=0; m < Os[0]; m++) {
        // Reposition index into input image based on requested output size
        if (mode == FULL) new_m = m ;
        else new_m = (m+dim_ker[0]-1);

        for (int n=0; n < Os[1]; n++) {  // loop over columns 
          %(type)s sum=0;

          // Sum over kernel, if index into image is out of bounds
          // fill with the value
          for (int j=0; j < dim_ker[0]; j++) {
            int ind0 = (new_m-j);

            if(mode==FULL){
              const %(type)s * idx_hvals=&hvals[j*dim_ker[1]];
              if(ind0 < 0 || ind0 >= dim_im[0]){
                if(fill_value!=0)
                  for (int k=0; k < dim_ker[1]; k++) {
                    sum+= idx_hvals[k] * fill_value;
                  }
              }else{
                //do the part where kernel is to the right of the img

                int k=0,max_k=max((int)(n-dim_im[1])+1,0);
                if(fill_value!=0){ 
                
                  for(k=0;k<max_k;k++){
                    sum+= idx_hvals[k]*fill_value;
                  }
                }else {k=max_k;}
                
                //do the part where the kernel is on the img
                max_k=min(n+1,(int)dim_ker[1]);
                const %(type)s * idx_in=&in[ind0*dim_im[1]];
                for (int ind1=n-k; k<max_k; k++,ind1--) {
                  sum+= idx_hvals[k] * idx_in[ind1];
                }
                //do the part to the left of the img
                if(fill_value!=0)
                  for(;k<dim_ker[1];k++) sum+= idx_hvals[k]*fill_value;
              }
            }else{
              const %(type)s* idx_in=&in[ind0*dim_im[1]]; //JB: should be dim_im[1] right? (was dim_im[0])
              const %(type)s* idx_hvals=&hvals[j*dim_ker[1]];
              int new_n = (n+dim_ker[1]-1);

              for (int k=0,last=new_n; k < dim_ker[1]; k++,last--) {
                sum+=idx_hvals[k]*idx_in[last];
              }
            }
          }//for j
          out[m*dim_zz[1]+n] %(affectation)s sum;
        }//for n
      }//for m
    }//for stack_size
    if (0 && (mode==FULL)){
      for (int i = 0; i < dim_zz[0]*dim_zz[1]; ++i) 
        std::cout << " " << out[i];
      std::cout << "\\n";
    }
  }//for n_kern
}//for b
Py_XDECREF(img2d);
Py_XDECREF(filtersflipped);
"""


#########  
#########  ConvOp c_code for valid mode (uses gemm)
#########

_conv_op_code_valid_gemm = """
int mode=-1,typenum=0, typenum_f=0;
PyArrayObject *ain1=NULL, *ain2=NULL, *img2d_arr=NULL;
const int NKERN = %(self_nkern)s;

int type_im=PyArray_TYPE(%(img2d)s);
int type_ker=PyArray_TYPE(%(filtersflipped)s);

npy_intp dim_zz[2]={%(self_outshp0)s,%(self_outshp1)s};
npy_intp dim_im[2]={%(self_imshp1)s,%(self_imshp2)s};
npy_intp dim_ker[2]={%(self_kshp0)s,%(self_kshp1)s};

PyArray_Dims img2d_shape;
npy_intp img2d_dim[4]={1,1,0,0};
img2d_shape.ptr=img2d_dim;
img2d_shape.len=4;

PyArray_Dims kerns_shape;
npy_intp kerns_dim[4]={1,1,0,0};
kerns_shape.ptr=kerns_dim;
kerns_shape.len=4;
PyObject *img2d=NULL, *contig;

if(%(img2d)s->nd==2){
  img2d_dim[3]=%(img2d)s->dimensions[1];
  img2d_dim[2]=%(img2d)s->dimensions[0];
}else if(%(img2d)s->nd==3){
  img2d_dim[3]=%(img2d)s->dimensions[2];
  img2d_dim[2]=%(img2d)s->dimensions[1];
  img2d_dim[0]=%(img2d)s->dimensions[0];
}else if(%(img2d)s->nd==4){
  img2d_dim[3]=%(img2d)s->dimensions[3];
  img2d_dim[2]=%(img2d)s->dimensions[2];
  img2d_dim[1]=%(img2d)s->dimensions[1];
  img2d_dim[0]=%(img2d)s->dimensions[0];
}else {
    PyErr_SetString(PyExc_ValueError, "img don't have a good shape");
    %(fail)s;
}

if(%(filtersflipped)s->nd==3){
  kerns_dim[3]=%(filtersflipped)s->dimensions[2];
  kerns_dim[2]=%(filtersflipped)s->dimensions[1];
  kerns_dim[0]=%(filtersflipped)s->dimensions[0];
}else if(%(filtersflipped)s->nd==4){
  kerns_dim[3]=%(filtersflipped)s->dimensions[3];
  kerns_dim[2]=%(filtersflipped)s->dimensions[2];
  kerns_dim[1]=%(filtersflipped)s->dimensions[1];
  kerns_dim[0]=%(filtersflipped)s->dimensions[0];
}else{
    PyErr_SetString(PyExc_ValueError, "kernel don't have a good shape");
    %(fail)s;
}
if (NKERN != kerns_dim[0])
{
    PyErr_SetString(PyExc_NotImplementedError, "nonsense nkern");
    %(fail)s;
}

img2d = PyArray_Newshape(%(img2d)s,&img2d_shape, PyArray_CORDER);
img2d_arr = (PyArrayObject*)img2d;
if ((img2d_arr->strides[3] != sizeof(%(type)s)) 
     || (img2d_arr->strides[2] != img2d_arr->dimensions[3]*sizeof(%(type)s))){
    contig = (PyObject*)(PyArray_GETCONTIGUOUS((PyArrayObject*)img2d));
    Py_DECREF(img2d);
    img2d = contig;
    if (!PyArray_ISCONTIGUOUS(img2d)){
        PyErr_SetString(PyExc_ValueError, "img2d isn't contiguous");
        %(fail)s;
    }
}
img2d_arr = (PyArrayObject*)img2d;

typenum = PyArray_ObjectType((PyObject*)%(img2d)s, 0);
typenum_f = PyArray_ObjectType((PyObject*)%(filtersflipped)s, 0);
if (typenum < 0) {PyErr_SetString(PyExc_ValueError, "Invalid type"); %(fail)s;}
if (typenum != typenum_f) {PyErr_SetString(PyExc_ValueError, "Input types must match"); %(fail)s;}

if (!img2d) {
    PyErr_SetString(PyExc_ValueError, "Null argument img2d");
    %(fail)s;
}
if ((!%(z)s)
  || *PyArray_DIMS(%(z)s)!=4
  ||(%(z)s->dimensions[0] != %(self_bsize)s)
  ||(%(z)s->dimensions[1] != %(self_nkern)s)
  ||(%(z)s->dimensions[2] != dim_zz[0])
  || (%(z)s->dimensions[3] != dim_zz[1])
  )
{
  if (%(z)s) Py_DECREF(%(z)s);
  npy_intp dims[4] = {0,0,0,0};
  dims[0]=%(self_bsize)s;
  dims[1]=%(self_nkern)s;
  dims[2]=dim_zz[0];
  dims[3]=dim_zz[1];
  %(z)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum,0);
}else{
  PyArray_FILLWBYTE((PyObject*)%(z)s,0);
}

int Os[2];
Os[0] = dim_im[0]-dim_ker[0]+1;
Os[1] = dim_im[1]-dim_ker[1]+1;

// allocate a temporary buffer for storing the inner product of each nth kernel row 
// with each row of an image
{
%(type)s * kbuf = (%(type)s *)malloc((Os[0] * NKERN + PyArray_Size((PyObject*)%(filtersflipped)s))* sizeof(%(type)s));
int kbufstride = NKERN;
%(type)s * myfilters = kbuf + Os[0] * NKERN;

//copy out filtersflipped into filters un-flipped format
//std::cerr << "__filling myfilters__\\n";
for(int i=0;i < kerns_dim[0];++i){
    for(int j=0;j < kerns_dim[1];++j){
        for(int k=0;k < kerns_dim[2];++k){
            for(int l=0;l < kerns_dim[3];++l){
                %(type)s * ff = ((%(filtersflipped)s)->nd == 3)
                    ? (%(type)s *)PyArray_GETPTR3(%(filtersflipped)s, i, kerns_dim[2]-1-k, kerns_dim[3]-1-l)
                    : (%(type)s *)PyArray_GETPTR4(%(filtersflipped)s, i, j, kerns_dim[2]-1-k, kerns_dim[3]-1-l);
                myfilters[i * (kerns_dim[1]*kerns_dim[2]*kerns_dim[3]) 
                          + j * (kerns_dim[2]*kerns_dim[3])
                          + k * (kerns_dim[3])
                          + l] = ff[0];
                //std::cerr << " " << ff[0];
            }
            //std::cerr << "\\n";
        }
        //std::cerr << "(end of stack/batch " <<j << "/" << i << "  ) \\n";
    }
}

//std::cerr << "-----new loop ----\\n";
for(int b=0;b< %(self_bsize)s;b++){
    for (int img_col = 0; img_col < Os[1]; ++img_col){
        for (int filter_row = 0; filter_row < kerns_dim[2]; ++filter_row){
            for (int stackidx = 0; stackidx < %(self_imshp0)s; ++stackidx){
                %(type)s * img_colview = 
                    (%(type)s *)(PyArray_GETPTR4(img2d, b, stackidx, filter_row, img_col));
                %(type)s * filter_rows = myfilters + stackidx * (kerns_dim[2]*kerns_dim[3]) +
                filter_row * kerns_dim[3];
                //std::cerr << "filterview offset: " << filter_rows - myfilters << "\\n";

                char N = 'N'; char T = 'T';
                int Nz0 = Os[0]; 
                int Nz1 = NKERN;
                int K = kerns_dim[3];
                double alpha = 1.0;
                double beta = stackidx ? 1.0 : 0.0;
                int imgview_stride = dim_im[1];
                int filter_rows_stride =kerns_dim[1]*kerns_dim[2]*kerns_dim[3];
                //remember, Fortran wants a column-major interpretation
                assert(img2d->strides[3] == sizeof(double));

                if (0){
                    std::cerr << "b " << b << " img_col " << img_col << " filterrow " << filter_row << " stackidx " <<stackidx << "\\n";
                    std::cerr << "colview (physical layout) stride: " << imgview_stride << "\\n";
                    for (int ii = 0; ii < Nz0; ++ii){
                        for (int jj = 0; jj < K; ++jj){
                            std::cerr << " " << img_colview[ii * imgview_stride + jj];
                        }
                        std::cerr << "\\n";
                    }
                    std::cerr << "filterview ("<<filter_row<<"'th rows) stride: " << filter_rows_stride << "\\n";
                    for (int ii = 0; ii < Nz1; ++ii){
                        for (int jj = 0; jj < K; ++jj){
                            std::cerr << " " << filter_rows[ii * filter_rows_stride + jj];
                        }
                        std::cerr << "\\n";
                    }

                    std::cerr << Nz1 << " " << Nz0 << " " << K << "\\n" ;
                }

                dgemm_(&T, &N, 
                    &Nz1, &Nz0, &K,
                    &alpha, 
                    filter_rows, &filter_rows_stride,
                    img_colview, &imgview_stride, 
                    &beta, kbuf, &kbufstride);

                if (0){
                    std::cerr << "z (logical layout) beta" << beta << "\\n";
                    for (int ii = 0; ii < Nz0; ++ii){
                        for (int jj = 0; jj < Nz1; ++jj){
                            std::cerr << " " << kbuf[ii * kbufstride + jj];
                        }
                        std::cerr << "\\n";
                    }
                }
            }
            // now kbuf the sum over the stack, put it into the outbuf
            for (int img_row = 0; img_row < Os[0]; ++img_row) {
                for (int kernel_idx = 0; kernel_idx < NKERN; ++kernel_idx) {
                    %(type)s * z_p =  (%(type)s *)PyArray_GETPTR4(%(z)s, b, kernel_idx, img_row, img_col);
                    if (0)
                    {
                        if (b >= %(z)s->dimensions[0]) %(fail)s;
                        if (kernel_idx >= %(z)s->dimensions[1]) %(fail)s;
                        if (img_row >= %(z)s->dimensions[2]) %(fail)s;
                        if (img_col >= %(z)s->dimensions[3]) %(fail)s;
                    }
                    z_p[0] += kbuf[img_row * kbufstride + kernel_idx];
                }
            }
        }
    }
}
free(kbuf);
}
Py_XDECREF(img2d);
"""


def gen_conv_code_unroll_batch(d,unroll_size=1):
    """ c_code for ConvOp that unroll the batch size loop
    """
    d["unroll_size"]=unroll_size
    def my_dup(st):
        s=""
        for i in range(unroll_size):
            d["unroll_iter"]=i
            s+=st%d
        return s
    ret = """
int mode=-1,typenum=0, typenum_f=0;
PyArrayObject *ain1=NULL, *ain2=NULL, *filtersflipped_arr=NULL, *img2d_arr=NULL;
const %(type)s fill_value = 0;

int type_im=PyArray_TYPE(%(img2d)s);
int type_ker=PyArray_TYPE(%(filtersflipped)s);

npy_intp dim_zz[2]={%(self_outshp0)s,%(self_outshp1)s};
npy_intp dim_im[2]={%(self_imshp1)s,%(self_imshp2)s};
npy_intp dim_ker[2]={%(self_kshp0)s,%(self_kshp1)s};

PyArray_Dims img2d_shape;
npy_intp img2d_dim[4]={1,1,0,0};
img2d_shape.ptr=img2d_dim;
img2d_shape.len=4;

PyArray_Dims kerns_shape;
npy_intp kerns_dim[4]={1,1,0,0};
kerns_shape.ptr=kerns_dim;
kerns_shape.len=4;
PyObject *img2d=NULL, *contig, *filtersflipped=NULL;
string s="%(self_out_mode)s";

if(%(img2d)s->nd==2){
  img2d_dim[3]=%(img2d)s->dimensions[1];
  img2d_dim[2]=%(img2d)s->dimensions[0];
}else if(%(img2d)s->nd==3){
  img2d_dim[3]=%(img2d)s->dimensions[2];
  img2d_dim[2]=%(img2d)s->dimensions[1];
  img2d_dim[0]=%(img2d)s->dimensions[0];
}else if(%(img2d)s->nd==4){
  img2d_dim[3]=%(img2d)s->dimensions[3];
  img2d_dim[2]=%(img2d)s->dimensions[2];
  img2d_dim[1]=%(img2d)s->dimensions[1];
  img2d_dim[0]=%(img2d)s->dimensions[0];
}else {
    PyErr_SetString(PyExc_ValueError, "img don't have a good shape");
    %(fail)s;
}

if(%(filtersflipped)s->nd==3){
  kerns_dim[3]=%(filtersflipped)s->dimensions[2];
  kerns_dim[2]=%(filtersflipped)s->dimensions[1];
  kerns_dim[0]=%(filtersflipped)s->dimensions[0];
}else if(%(filtersflipped)s->nd==4){
  kerns_dim[3]=%(filtersflipped)s->dimensions[3];
  kerns_dim[2]=%(filtersflipped)s->dimensions[2];
  kerns_dim[1]=%(filtersflipped)s->dimensions[1];
  kerns_dim[0]=%(filtersflipped)s->dimensions[0];
}else{
    PyErr_SetString(PyExc_ValueError, "kernel don't have a good shape");
    %(fail)s;
}

img2d = PyArray_Newshape(%(img2d)s,&img2d_shape, PyArray_CORDER);
img2d_arr = (PyArrayObject*)img2d;
if ((img2d_arr->strides[3] != sizeof(%(type)s)) 
     || (img2d_arr->strides[2] != img2d_arr->dimensions[3]*sizeof(%(type)s))){
    contig = (PyObject*)(PyArray_GETCONTIGUOUS((PyArrayObject*)img2d));
    Py_DECREF(img2d);
    img2d = contig;
    if (!PyArray_ISCONTIGUOUS(img2d)){
        PyErr_SetString(PyExc_ValueError, "img2d isn't contiguous");
        %(fail)s;
    }
}
img2d_arr = (PyArrayObject*)img2d;

filtersflipped = PyArray_Newshape(%(filtersflipped)s,&kerns_shape, PyArray_CORDER);
filtersflipped_arr = (PyArrayObject*)filtersflipped;
if ((filtersflipped_arr->strides[3] != sizeof(%(type)s)) 
     || (filtersflipped_arr->strides[2] != filtersflipped_arr->dimensions[3]*sizeof(%(type)s))){
    contig = (PyObject*)(PyArray_GETCONTIGUOUS((PyArrayObject*)filtersflipped));
    Py_DECREF(filtersflipped);
    filtersflipped = contig;
    if (!PyArray_ISCONTIGUOUS(filtersflipped)){
        PyErr_SetString(PyExc_ValueError, "filtersflipped isn't contiguous");
        %(fail)s;
    }
}
filtersflipped_arr = (PyArrayObject*)filtersflipped;

if(s=="valid") mode=0;
else if(s=="full") mode=2;
else {PyErr_SetString(PyExc_ValueError, "invalid mode, only full and valid are supported"); %(fail)s;};
typenum = PyArray_ObjectType((PyObject*)%(img2d)s, 0);
typenum_f = PyArray_ObjectType((PyObject*)%(filtersflipped)s, 0);
if (typenum < 0) {PyErr_SetString(PyExc_ValueError, "Invalid type"); %(fail)s;}
if (typenum != typenum_f) {PyErr_SetString(PyExc_ValueError, "Input types must match"); %(fail)s;}

if (!img2d) %(fail)s;
if (!filtersflipped) %(fail)s;
if ((!%(z)s)
  || *PyArray_DIMS(%(z)s)!=4
  ||(%(z)s->dimensions[0] != %(self_bsize)s)
  ||(%(z)s->dimensions[1] != %(self_nkern)s)
  ||(%(z)s->dimensions[2] != dim_zz[0])
  || (%(z)s->dimensions[3] != dim_zz[1])
  )
{
  if (%(z)s) Py_DECREF(%(z)s);
  npy_intp dims[4] = {0,0,0,0};
  if(!dims) %(fail)s;
  dims[0]=%(self_bsize)s;
  dims[1]=%(self_nkern)s;
  dims[2]=dim_zz[0];
  dims[3]=dim_zz[1];
  %(z)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum,0);
}else{
  //PyArray_FILLWBYTE((PyObject*)%(z)s,0);
}

int Os[2];
if (mode == FULL) {Os[0] = dim_im[0]+dim_ker[0]-1; Os[1] = dim_im[1]+dim_ker[1]-1;}
else {Os[0] = dim_im[0]-dim_ker[0]+1; Os[1] = dim_im[1]-dim_ker[1]+1;}
for(int b=0;b< %(self_bsize)s ;b+=%(unroll_size)s){
  for(int n_kern=0;n_kern<%(self_nkern)s;n_kern++){

    //assertions
    if (%(z)s->strides[0] != %(z)s->dimensions[1] *%(z)s->dimensions[2] *%(z)s->dimensions[3] * sizeof(%(type)s)) %(fail)s;
    if (%(z)s->strides[1] != %(z)s->dimensions[2] * %(z)s->dimensions[3] * sizeof(%(type)s)) %(fail)s;
    if (%(z)s->strides[2] != %(z)s->dimensions[3] * sizeof(%(type)s)) %(fail)s;
    if (%(z)s->strides[3] != sizeof(%(type)s)) %(fail)s;
"""%d
    ret+=my_dup("%(type)s * __restrict__ out%(unroll_iter)s=(%(type)s *)(PyArray_GETPTR2(%(z)s,b+%(unroll_iter)s,n_kern));\n")
    ret+=my_dup("for (int i = 0; i < dim_zz[0]*dim_zz[1]; ++i) out%(unroll_iter)s[i] = 0;")
    ret+="""
    for(int stack_size=0;stack_size<%(self_imshp0)s;stack_size++){
"""%d
    ret+=my_dup("const %(type)s * __restrict__ in%(unroll_iter)d=(%(type)s *)(PyArray_GETPTR2(img2d,b+%(unroll_iter)s,stack_size));\n")
    ret+="""
      const %(type)s * __restrict__ hvals=(%(type)s *)(PyArray_GETPTR2(filtersflipped,n_kern,stack_size));

      int new_m;

      for (int m=0; m < Os[0]; m++) {
        // Reposition index into input image based on requested output size
        if (mode == FULL) new_m = m ;
        else new_m = (m+dim_ker[0]-1);

        for (int n=0; n < Os[1]; n++) {  // loop over columns 
        """%d
    ret+=my_dup("%(type)s sum%(unroll_iter)s=0;\n")
    ret+="""

          // Sum over kernel, if index into image is out of bounds
          // fill with the value
          for (int j=0; j < dim_ker[0]; j++) {
            int ind0 = (new_m-j);

            if(mode==FULL){
              const %(type)s * idx_hvals=&hvals[j*dim_ker[1]];
              if(ind0 < 0 || ind0 >= dim_im[0]){
                if(fill_value!=0)
                  for (int k=0; k < dim_ker[1]; k++) {
                    %(type)s tmp = idx_hvals[k] * fill_value;
"""%d
    ret+=my_dup("sum%(unroll_iter)s += tmp;\n")
    ret+="""
                  }
              }else{
                //do the part where kernel is to the right of the img

                int k=0,max_k=max((int)(n-dim_im[1])+1,0);
                if(fill_value!=0){ 
                
                  for(k=0;k<max_k;k++){
                    %(type)s tmp = idx_hvals[k] * fill_value;
"""%d
    ret+=my_dup("sum%(unroll_iter)s += tmp;\n")
    ret+="""
                  }
                }else {k=max_k;}
                
                //do the part where the kernel is on the img
                max_k=min(n+1,(int)dim_ker[1]);
"""%d
    ret+=my_dup("const %(type)s * idx_in%(unroll_iter)s=&in%(unroll_iter)s[ind0*dim_im[1]];\n")
    ret+="""
                for (int ind1=n-k; k<max_k; k++,ind1--) {
"""%d
    ret+=my_dup("sum%(unroll_iter)s+= idx_hvals[k] * idx_in%(unroll_iter)s[ind1];\n")
    ret+="""
                }
                //do the part to the left of the img
                if(fill_value!=0)
                  for(;k<dim_ker[1];k++){
                    %(type)s tmp = idx_hvals[k] * fill_value;
"""%d
    ret+=my_dup("sum%(unroll_iter)s += tmp;\n")
    ret+="""
                  }
              }
            }else{
"""%d
    ret+=my_dup("const %(type)s* idx_in%(unroll_iter)s=&in%(unroll_iter)s[ind0*dim_im[1]];\n")
    ret+="""
              const %(type)s* idx_hvals=&hvals[j*dim_ker[1]];
              int new_n = (n+dim_ker[1]-1);

              for (int k=0,last=new_n; k < dim_ker[1]; k++,last--) {
"""%d
    ret+=my_dup("sum%(unroll_iter)s+=idx_hvals[k]*idx_in%(unroll_iter)s[last];\n")
    ret+="""
              }
            }

          }//for j
"""%d
    ret+=my_dup("out%(unroll_iter)s[m*dim_zz[1]+n] %(affectation)s sum%(unroll_iter)s;\n")
#        ret+=my_dup("cout<<sum%(unroll_iter)s<<endl;")
    ret+="""
        }//for n
      }//for m
    }//for stack_size
    if (0 && (mode==FULL)){
      for (int i = 0; i < dim_zz[0]*dim_zz[1]; ++i) 
        std::cout << " " << out0[i];
      std::cout << "\\n";
    }
  }//for n_kern
}//for b
Py_XDECREF(img2d);
Py_XDECREF(filtersflipped);
"""
    return ret



def gen_conv_code_unroll_kern(d,unroll_size=1):
    """ c_code for ConvOp that unroll the batch size loop
    """
    d["unroll_size"]=unroll_size
    def my_dup(st):
        s=""
        for i in range(unroll_size):
            d["unroll_iter"]=i
            s+=st%d
        return s
    ret = """
int mode=-1,typenum=0, typenum_f=0;
PyArrayObject *ain1=NULL, *ain2=NULL, *filtersflipped_arr=NULL, *img2d_arr=NULL;
const %(type)s fill_value = 0;

int type_im=PyArray_TYPE(%(img2d)s);
int type_ker=PyArray_TYPE(%(filtersflipped)s);

npy_intp dim_zz[2]={%(self_outshp0)s,%(self_outshp1)s};
npy_intp dim_im[2]={%(self_imshp1)s,%(self_imshp2)s};
npy_intp dim_ker[2]={%(self_kshp0)s,%(self_kshp1)s};

PyArray_Dims img2d_shape;
npy_intp img2d_dim[4]={1,1,0,0};
img2d_shape.ptr=img2d_dim;
img2d_shape.len=4;

PyArray_Dims kerns_shape;
npy_intp kerns_dim[4]={1,1,0,0};
kerns_shape.ptr=kerns_dim;
kerns_shape.len=4;
PyObject *img2d=NULL, *contig, *filtersflipped=NULL;
string s="%(self_out_mode)s";

if(%(img2d)s->nd==2){
  img2d_dim[3]=%(img2d)s->dimensions[1];
  img2d_dim[2]=%(img2d)s->dimensions[0];
}else if(%(img2d)s->nd==3){
  img2d_dim[3]=%(img2d)s->dimensions[2];
  img2d_dim[2]=%(img2d)s->dimensions[1];
  img2d_dim[0]=%(img2d)s->dimensions[0];
}else if(%(img2d)s->nd==4){
  img2d_dim[3]=%(img2d)s->dimensions[3];
  img2d_dim[2]=%(img2d)s->dimensions[2];
  img2d_dim[1]=%(img2d)s->dimensions[1];
  img2d_dim[0]=%(img2d)s->dimensions[0];
}else {
    PyErr_SetString(PyExc_ValueError, "img don't have a good shape");
    %(fail)s;
}

if(%(filtersflipped)s->nd==3){
  kerns_dim[3]=%(filtersflipped)s->dimensions[2];
  kerns_dim[2]=%(filtersflipped)s->dimensions[1];
  kerns_dim[0]=%(filtersflipped)s->dimensions[0];
}else if(%(filtersflipped)s->nd==4){
  kerns_dim[3]=%(filtersflipped)s->dimensions[3];
  kerns_dim[2]=%(filtersflipped)s->dimensions[2];
  kerns_dim[1]=%(filtersflipped)s->dimensions[1];
  kerns_dim[0]=%(filtersflipped)s->dimensions[0];
}else{
    PyErr_SetString(PyExc_ValueError, "kernel don't have a good shape");
    %(fail)s;
}

img2d = PyArray_Newshape(%(img2d)s,&img2d_shape, PyArray_CORDER);
img2d_arr = (PyArrayObject*)img2d;
if ((img2d_arr->strides[3] != sizeof(%(type)s)) 
     || (img2d_arr->strides[2] != img2d_arr->dimensions[3]*sizeof(%(type)s))){
    contig = (PyObject*)(PyArray_GETCONTIGUOUS((PyArrayObject*)img2d));
    Py_DECREF(img2d);
    img2d = contig;
    if (!PyArray_ISCONTIGUOUS(img2d)){
        PyErr_SetString(PyExc_ValueError, "img2d isn't contiguous");
        %(fail)s;
    }
}
img2d_arr = (PyArrayObject*)img2d;

filtersflipped = PyArray_Newshape(%(filtersflipped)s,&kerns_shape, PyArray_CORDER);
filtersflipped_arr = (PyArrayObject*)filtersflipped;
if ((filtersflipped_arr->strides[3] != sizeof(%(type)s)) 
     || (filtersflipped_arr->strides[2] != filtersflipped_arr->dimensions[3]*sizeof(%(type)s))){
    contig = (PyObject*)(PyArray_GETCONTIGUOUS((PyArrayObject*)filtersflipped));
    Py_DECREF(filtersflipped);
    filtersflipped = contig;
    if (!PyArray_ISCONTIGUOUS(filtersflipped)){
        PyErr_SetString(PyExc_ValueError, "filtersflipped isn't contiguous");
        %(fail)s;
    }
}
filtersflipped_arr = (PyArrayObject*)filtersflipped;

if(s=="valid") mode=0;
else if(s=="full") mode=2;
else {PyErr_SetString(PyExc_ValueError, "invalid mode, only full and valid are supported"); %(fail)s;};
typenum = PyArray_ObjectType((PyObject*)%(img2d)s, 0);
typenum_f = PyArray_ObjectType((PyObject*)%(filtersflipped)s, 0);
if (typenum < 0) {PyErr_SetString(PyExc_ValueError, "Invalid type"); %(fail)s;}
if (typenum != typenum_f) {PyErr_SetString(PyExc_ValueError, "Input types must match"); %(fail)s;}

if (!img2d) %(fail)s;
if (!filtersflipped) %(fail)s;
if ((!%(z)s)
  || *PyArray_DIMS(%(z)s)!=4
  ||(%(z)s->dimensions[0] != %(self_bsize)s)
  ||(%(z)s->dimensions[1] != %(self_nkern)s)
  ||(%(z)s->dimensions[2] != dim_zz[0])
  || (%(z)s->dimensions[3] != dim_zz[1])
  )
{
  if (%(z)s) Py_DECREF(%(z)s);
  npy_intp dims[4] = {0,0,0,0};
  if(!dims) %(fail)s;
  dims[0]=%(self_bsize)s;
  dims[1]=%(self_nkern)s;
  dims[2]=dim_zz[0];
  dims[3]=dim_zz[1];
  %(z)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum,0);
}else{
  //PyArray_FILLWBYTE((PyObject*)%(z)s,0);
}

int Os[2];
if (mode == FULL) {Os[0] = dim_im[0]+dim_ker[0]-1; Os[1] = dim_im[1]+dim_ker[1]-1;}
else {Os[0] = dim_im[0]-dim_ker[0]+1; Os[1] = dim_im[1]-dim_ker[1]+1;}

for(int b=0;b< %(self_bsize)s;b++){
  for(int n_kern=0;n_kern<%(self_nkern)s;n_kern+=%(unroll_size)s){

    //assertions
    if (%(z)s->strides[0] != %(z)s->dimensions[1] *%(z)s->dimensions[2] *%(z)s->dimensions[3] * sizeof(%(type)s)) %(fail)s;
    if (%(z)s->strides[1] != %(z)s->dimensions[2] * %(z)s->dimensions[3] * sizeof(%(type)s)) %(fail)s;
    if (%(z)s->strides[2] != %(z)s->dimensions[3] * sizeof(%(type)s)) %(fail)s;
    if (%(z)s->strides[3] != sizeof(%(type)s)) %(fail)s;
"""%d
    ret+=my_dup("%(type)s * __restrict__ out%(unroll_iter)s=(%(type)s *)(PyArray_GETPTR2(%(z)s,b,n_kern+%(unroll_iter)s));")
    ret+=my_dup("for (int i = 0; i < dim_zz[0]*dim_zz[1]; ++i) out%(unroll_iter)s[i] = 0;")
    ret+="""

    for(int stack_size=0;stack_size<%(self_imshp0)s;stack_size++){

      const %(type)s * __restrict__ in=(%(type)s *)(PyArray_GETPTR2(img2d,b,stack_size));
"""%d
    ret+=my_dup("const %(type)s * __restrict__ hvals%(unroll_iter)s=(%(type)s *)(PyArray_GETPTR2(filtersflipped,n_kern+%(unroll_iter)s,stack_size));")
    ret+="""

      int new_m;

      for (int m=0; m < Os[0]; m++) {
        // Reposition index into input image based on requested output size
        if (mode == FULL) new_m = m ;
        else new_m = (m+dim_ker[0]-1);

        for (int n=0; n < Os[1]; n++) {  // loop over columns 
"""%d
    ret+=my_dup("%(type)s sum%(unroll_iter)s=0;")
    ret+="""

          // Sum over kernel, if index into image is out of bounds
          // fill with the value
          for (int j=0; j < dim_ker[0]; j++) {
            int ind0 = (new_m-j);

            if(mode==FULL){
"""%d
    ret+=my_dup("const %(type)s * idx_hvals%(unroll_iter)s=&hvals%(unroll_iter)s[j*dim_ker[1]];")
    ret+="""
              if(ind0 < 0 || ind0 >= dim_im[0]){
                if(fill_value!=0)
                  for (int k=0; k < dim_ker[1]; k++) {
"""%d
    ret+=my_dup("sum%(unroll_iter)s += idx_hvals%(unroll_iter)s[k] * fill_value;")
    ret+="""
                  }
              }else{
                //do the part where kernel is to the right of the img

                int k=0,max_k=max((int)(n-dim_im[1])+1,0);
                if(fill_value!=0){ 
                
                  for(k=0;k<max_k;k++){
"""%d
    ret+=my_dup("sum%(unroll_iter)s += idx_hvals%(unroll_iter)s[k]*fill_value;")

    ret+="""
                  }
                }else {k=max_k;}
                
                //do the part where the kernel is on the img
                max_k=min(n+1,(int)dim_ker[1]);
                const %(type)s * idx_in=&in[ind0*dim_im[1]];
                for (int ind1=n-k; k<max_k; k++,ind1--) {
"""%d
    ret+=my_dup("sum%(unroll_iter)s += idx_hvals%(unroll_iter)s[k] * idx_in[ind1];")
    ret+="""
                }
                //do the part to the left of the img
                if(fill_value!=0)
                  for(;k<dim_ker[1];k++){
"""%d
    ret+=my_dup("sum%(unroll_iter)s+= idx_hvals%(unroll_iter)s[k]*fill_value;")
    ret+="""
                  }
              }
            }else{
              const %(type)s* idx_in=&in[ind0*dim_im[1]];
"""%d
    ret+=my_dup("const %(type)s* idx_hvals%(unroll_iter)s=&hvals%(unroll_iter)s[j*dim_ker[1]];")
    ret+="""
              int new_n = (n+dim_ker[1]-1);

              for (int k=0,last=new_n; k < dim_ker[1]; k++,last--) {
"""%d
    ret+=my_dup("sum%(unroll_iter)s += idx_hvals%(unroll_iter)s[k]*idx_in[last];")
    ret+="""
              }
            }
          }//for j
"""%d
    ret+=my_dup("out%(unroll_iter)s[m*dim_zz[1]+n] %(affectation)s sum%(unroll_iter)s;")
    ret+="""
        }//for n
      }//for m
    }//for stack_size
  }//for n_kern
}//for b
Py_XDECREF(img2d);
Py_XDECREF(filtersflipped);
"""%d
    return ret



def gen_conv_code_unroll_batch_kern(d,unroll_bsize=1, unroll_ksize=1):
    """ c_code for ConvOp that unroll the batch size loop
    """
    d["unroll_bsize"]=unroll_bsize
    d["unroll_ksize"]=unroll_ksize
    def my_dup(st,size):
        s=""
        for i in range(size):
            d["unroll_iter"]=i
            s+=st%d
        return s+"\n"
    def my_dup2(st):
        s=""
        iter=0
        for i in range(unroll_bsize):
            d["unroll_biter"]=i
            for j in range(unroll_ksize):
                d["unroll_kiter"]=j
                d["unroll_iter"]=iter
                iter+=1
                s+=st%d
        return s+"\n"
    ret = """
int mode=-1,typenum=0, typenum_f=0;
PyArrayObject *ain1=NULL, *ain2=NULL, *filtersflipped_arr=NULL, *img2d_arr=NULL;
const %(type)s fill_value = 0;

int type_im=PyArray_TYPE(%(img2d)s);
int type_ker=PyArray_TYPE(%(filtersflipped)s);

npy_intp dim_zz[2]={%(self_outshp0)s,%(self_outshp1)s};
npy_intp dim_im[2]={%(self_imshp1)s,%(self_imshp2)s};
npy_intp dim_ker[2]={%(self_kshp0)s,%(self_kshp1)s};

PyArray_Dims img2d_shape;
npy_intp img2d_dim[4]={1,1,0,0};
img2d_shape.ptr=img2d_dim;
img2d_shape.len=4;

PyArray_Dims kerns_shape;
npy_intp kerns_dim[4]={1,1,0,0};
kerns_shape.ptr=kerns_dim;
kerns_shape.len=4;
PyObject *img2d=NULL, *contig, *filtersflipped=NULL;
string s="%(self_out_mode)s";

if(%(img2d)s->nd==2){
  img2d_dim[3]=%(img2d)s->dimensions[1];
  img2d_dim[2]=%(img2d)s->dimensions[0];
}else if(%(img2d)s->nd==3){
  img2d_dim[3]=%(img2d)s->dimensions[2];
  img2d_dim[2]=%(img2d)s->dimensions[1];
  img2d_dim[0]=%(img2d)s->dimensions[0];
}else if(%(img2d)s->nd==4){
  img2d_dim[3]=%(img2d)s->dimensions[3];
  img2d_dim[2]=%(img2d)s->dimensions[2];
  img2d_dim[1]=%(img2d)s->dimensions[1];
  img2d_dim[0]=%(img2d)s->dimensions[0];
}else {
    PyErr_SetString(PyExc_ValueError, "img don't have a good shape");
    %(fail)s;
}

if(%(filtersflipped)s->nd==3){
  kerns_dim[3]=%(filtersflipped)s->dimensions[2];
  kerns_dim[2]=%(filtersflipped)s->dimensions[1];
  kerns_dim[0]=%(filtersflipped)s->dimensions[0];
}else if(%(filtersflipped)s->nd==4){
  kerns_dim[3]=%(filtersflipped)s->dimensions[3];
  kerns_dim[2]=%(filtersflipped)s->dimensions[2];
  kerns_dim[1]=%(filtersflipped)s->dimensions[1];
  kerns_dim[0]=%(filtersflipped)s->dimensions[0];
}else{
    PyErr_SetString(PyExc_ValueError, "kernel don't have a good shape");
    %(fail)s;
}

img2d = PyArray_Newshape(%(img2d)s,&img2d_shape, PyArray_CORDER);
img2d_arr = (PyArrayObject*)img2d;
if ((img2d_arr->strides[3] != sizeof(%(type)s)) 
     || (img2d_arr->strides[2] != img2d_arr->dimensions[3]*sizeof(%(type)s))){
    contig = (PyObject*)(PyArray_GETCONTIGUOUS((PyArrayObject*)img2d));
    Py_DECREF(img2d);
    img2d = contig;
    if (!PyArray_ISCONTIGUOUS(img2d)){
        PyErr_SetString(PyExc_ValueError, "img2d isn't contiguous");
        %(fail)s;
    }
}
img2d_arr = (PyArrayObject*)img2d;

filtersflipped = PyArray_Newshape(%(filtersflipped)s,&kerns_shape, PyArray_CORDER);
filtersflipped_arr = (PyArrayObject*)filtersflipped;
if ((filtersflipped_arr->strides[3] != sizeof(%(type)s)) 
     || (filtersflipped_arr->strides[2] != filtersflipped_arr->dimensions[3]*sizeof(%(type)s))){
    contig = (PyObject*)(PyArray_GETCONTIGUOUS((PyArrayObject*)filtersflipped));
    Py_DECREF(filtersflipped);
    filtersflipped = contig;
    if (!PyArray_ISCONTIGUOUS(filtersflipped)){
        PyErr_SetString(PyExc_ValueError, "filtersflipped isn't contiguous");
        %(fail)s;
    }
}
filtersflipped_arr = (PyArrayObject*)filtersflipped;

if(s=="valid") mode=0;
else if(s=="full") mode=2;
else {PyErr_SetString(PyExc_ValueError, "invalid mode, only full and valid are supported"); %(fail)s;};
typenum = PyArray_ObjectType((PyObject*)%(img2d)s, 0);
typenum_f = PyArray_ObjectType((PyObject*)%(filtersflipped)s, 0);
if (typenum < 0) {PyErr_SetString(PyExc_ValueError, "Invalid type"); %(fail)s;}
if (typenum != typenum_f) {PyErr_SetString(PyExc_ValueError, "Input types must match"); %(fail)s;}

if (!img2d) %(fail)s;
if (!filtersflipped) %(fail)s;
if ((!%(z)s)
  || *PyArray_DIMS(%(z)s)!=4
  ||(%(z)s->dimensions[0] != %(self_bsize)s)
  ||(%(z)s->dimensions[1] != %(self_nkern)s)
  ||(%(z)s->dimensions[2] != dim_zz[0])
  || (%(z)s->dimensions[3] != dim_zz[1])
  )
{
  if (%(z)s) Py_DECREF(%(z)s);
  npy_intp dims[4] = {0,0,0,0};
  if(!dims) %(fail)s;
  dims[0]=%(self_bsize)s;
  dims[1]=%(self_nkern)s;
  dims[2]=dim_zz[0];
  dims[3]=dim_zz[1];
  %(z)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum,0);
}else{
  //PyArray_FILLWBYTE((PyObject*)%(z)s,0);
}

int Os[2];
if (mode == FULL) {Os[0] = dim_im[0]+dim_ker[0]-1; Os[1] = dim_im[1]+dim_ker[1]-1;}
else {Os[0] = dim_im[0]-dim_ker[0]+1; Os[1] = dim_im[1]-dim_ker[1]+1;}
for(int b=0;b< %(self_bsize)s ;b+=%(unroll_bsize)s){
  for(int n_kern=0;n_kern<%(self_nkern)s;n_kern+=%(unroll_ksize)s){

    //assertions
    if (%(z)s->strides[0] != %(z)s->dimensions[1] *%(z)s->dimensions[2] *%(z)s->dimensions[3] * sizeof(%(type)s)) %(fail)s;
    if (%(z)s->strides[1] != %(z)s->dimensions[2] * %(z)s->dimensions[3] * sizeof(%(type)s)) %(fail)s;
    if (%(z)s->strides[2] != %(z)s->dimensions[3] * sizeof(%(type)s)) %(fail)s;
    if (%(z)s->strides[3] != sizeof(%(type)s)) %(fail)s;
"""%d
    ret+=my_dup2("%(type)s * __restrict__ out%(unroll_iter)s=(%(type)s *)(PyArray_GETPTR2(%(z)s,b+%(unroll_biter)s,n_kern+%(unroll_kiter)s));")
    ret+=my_dup("for (int i = 0; i < dim_zz[0]*dim_zz[1]; ++i) out%(unroll_iter)s[i] = 0;",unroll_bsize*unroll_ksize)
    ret+="""
    for(int stack_size=0;stack_size<%(self_imshp0)s;stack_size++){
"""%d
    ret+=my_dup("const %(type)s * __restrict__ in%(unroll_iter)d=(%(type)s *)(PyArray_GETPTR2(img2d,b+%(unroll_iter)s,stack_size));", unroll_bsize)
    ret+=my_dup("const %(type)s * __restrict__ hvals%(unroll_iter)s=(%(type)s *)(PyArray_GETPTR2(filtersflipped,n_kern+%(unroll_iter)s,stack_size));",unroll_ksize)
    ret+="""

      int new_m;

      for (int m=0; m < Os[0]; m++) {
        // Reposition index into input image based on requested output size
        if (mode == FULL) new_m = m ;
        else new_m = (m+dim_ker[0]-1);

        for (int n=0; n < Os[1]; n++) {  // loop over columns 
        """%d
    ret+=my_dup("%(type)s sum%(unroll_iter)s=0;", unroll_bsize*unroll_ksize)
    ret+="""

          // Sum over kernel, if index into image is out of bounds
          // fill with the value
          for (int j=0; j < dim_ker[0]; j++) {
            int ind0 = (new_m-j);

            if(mode==FULL){
"""%d
    ret+=my_dup("const %(type)s * idx_hvals%(unroll_iter)s=&hvals%(unroll_iter)s[j*dim_ker[1]];",unroll_ksize)
    ret+="""
              if(ind0 < 0 || ind0 >= dim_im[0]){
                if(fill_value!=0)
                  for (int k=0; k < dim_ker[1]; k++) {
"""%d
    ret+=my_dup2("sum%(unroll_iter)s += idx_hvals%(unroll_kiter)s[k] * fill_value;")
    ret+="""
                  }
              }else{
                //do the part where kernel is to the right of the img

                int k=0,max_k=max((int)(n-dim_im[1])+1,0);
                if(fill_value!=0){ 
                
                  for(k=0;k<max_k;k++){
"""%d
    ret+=my_dup2("sum%(unroll_iter)s += idx_hvals%(unroll_kiter)s[k] * fill_value;")
    ret+="""
                  }
                }else {k=max_k;}
                
                //do the part where the kernel is on the img
                max_k=min(n+1,(int)dim_ker[1]);
"""%d
    ret+=my_dup("const %(type)s * idx_in%(unroll_iter)s=&in%(unroll_iter)s[ind0*dim_im[1]];", unroll_bsize)
    ret+="""
                for (int ind1=n-k; k<max_k; k++,ind1--) {

"""%d
    ret+=my_dup2("sum%(unroll_iter)s+= idx_hvals%(unroll_kiter)s[k] * idx_in%(unroll_biter)s[ind1];")
    ret+="""
                }
                //do the part to the left of the img
                if(fill_value!=0)
                  for(;k<dim_ker[1];k++){
"""%d
    ret+=my_dup2("sum%(unroll_iter)s += idx_hvals%(unroll_kiter)s[k] * fill_value;")
    ret+="""
                  }
              }
            }else{
"""%d
    ret+=my_dup("const %(type)s* idx_in%(unroll_iter)s=&in%(unroll_iter)s[ind0*dim_im[1]];", unroll_bsize)
    ret+=my_dup("const %(type)s* idx_hvals%(unroll_iter)s=&hvals%(unroll_iter)s[j*dim_ker[1]];",unroll_ksize)
    ret+="""
              int new_n = (n+dim_ker[1]-1);

              for (int k=0,last=new_n; k < dim_ker[1]; k++,last--) {
"""%d
    ret+=my_dup2("sum%(unroll_iter)s+=idx_hvals%(unroll_kiter)s[k]*idx_in%(unroll_biter)s[last];")
    ret+="""
              }
            }

          }//for j
"""%d
#    ret+=my_dup("out%(unroll_iter)s[m*dim_zz[1]+n] %(affectation)s sum%(unroll_iter)s;", unroll_bsize)
    ret+=my_dup("out%(unroll_iter)s[m*dim_zz[1]+n] %(affectation)s sum%(unroll_iter)s;", unroll_bsize*unroll_ksize)
#        ret+=my_dup("cout<<sum%(unroll_iter)s<<endl;",unroll_bsize)
    ret+="""
        }//for n
      }//for m
    }//for stack_size
  }//for n_kern
}//for b
Py_XDECREF(img2d);
Py_XDECREF(filtersflipped);
"""
    return ret
