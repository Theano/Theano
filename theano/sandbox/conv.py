import numpy as N
import theano
import theano.tensor as T
from theano import gof, Op, tensor
from scipy.signal.signaltools import  _valfrommode, _bvalfromboundary
from scipy.signal.sigtools import _convolve2d
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

    def __init__(self, imshp, kshp, nkern, bsize, dx, dy, output_mode='valid'):
        if len(imshp)==2:
            self.imshp = (1,)+imshp
        elif len(imshp)==3:
            self.imshp = imshp
        else:
            raise Exception("bad len for imshp")
        self.kshp = kshp
        self.nkern = nkern
        self.bsize=bsize
        self.dx=dx
        self.dy=dy
        if self.dx!=1 or self.dy!=1:
            print "Warning, dx!=1 or dy!=1 only supported in python mode!"
            raise NotImplementedError()
        self.out_mode = output_mode
        if not self.out_mode in ["valid", "full"]:
            raise Exception("Mode %s not implemented"%self.out_mode)
        self.fulloutshp = N.array(self.imshp[1:]) - N.array(self.kshp) + 1 \
            if self.out_mode=='valid'\
            else N.array(self.imshp[1:]) + N.array(self.kshp) - 1

        assert ((N.array(self.imshp[1:])-self.kshp)>=0).all()
        assert N.prod(self.fulloutshp)>0

#    def __eq__(self, other):
#        raise Error("Not implemented")

#    def __hash__(self):
#        raise Error("Not implemented")

    def make_node(self, inputs, kerns):
        #all kernels must have the same shape!
        #output_mode only valid and full are supported!

        self.outshp = getFilterOutShp(self.imshp, self.kshp, (self.dx,self.dy), self.out_mode)
        self.dtype = inputs.dtype
        assert kerns.dtype==self.dtype
  
        # TODO: find a way to make ConvOp work for N-D (after NIPS09)
        outdim = kerns.ndim
        output = tensor.tensor(dtype=self.dtype, broadcastable=[False]*outdim);

        return gof.Apply(self, [inputs, kerns], [output])

    def perform(self,node, (img2d, filtersflipped), (z,)):
        """
        By default if len(img2d.shape)==3, we
        """
        if z[0] is None:
            z[0] = N.zeros((self.bsize,)+(self.nkern,)+tuple(self.fulloutshp))
        zz=z[0]
        val = _valfrommode(self.out_mode)
        bval = _bvalfromboundary('fill')
        if len(img2d.shape)==2 and self.imshp[0]==1 and self.bsize==1:
            img2d = img2d.reshape((1,1)+img2d.shape)
        elif len(img2d.shape)==3 and self.imshp[0]==1 and self.bsize!=1:
            img2d = img2d.reshape((img2d.shape[0],)+(1,)+img2d.shape[1:])            
        elif len(img2d.shape)==3:
            img2d = img2d.reshape((1,)+(img2d.shape[0],)+img2d.shape[1:])
        elif len(img2d.shape)==3 and self.imshp[0]==1 and self.bsize==1:
            img2d = img2d.reshape((1,1)+img2d.shape[1:])
        elif len(img2d.shape)!=4: raise Exception("bad img2d shape.")
        
        if len(filtersflipped.shape)==3 and self.imshp[0]==1:
            assert self.imshp[0]==1
            filtersflipped = filtersflipped.reshape((filtersflipped.shape[0],)+(1,)+filtersflipped.shape[1:])
        elif len(filtersflipped.shape)!=4: raise Exception("Bad filtersflipped shape")

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
        print '************GRAD**************'
        print 'self.outshp = ', self.outshp

        ####### Determine gradient on kernels ########
        if inputs.ndim == 3:
            print 'xxxxxx self.imshp = ', self.imshp
            print 'inputs.broadcastable = ', inputs.broadcastable
            print 'inputs.ndim = ', inputs.ndim
            img = tensor.shape_padleft(inputs,1)

        img = tensor.DimShuffle(inputs.broadcastable, (1,0,2,3))(inputs)
        imshp = N.hstack((self.bsize, self.imshp[1:]))
        bsize = self.imshp[0]

        nkern = self.nkern
        filters = tensor.DimShuffle(gz.broadcastable, (1,0,2,3))(gz)
        filters = filters[:,:,::-1,::-1]

        kshp  = self.outshp[::-1]

        print kshp, imshp

        mode = self.out_mode
        dw = ConvOp(imshp, kshp, nkern, bsize, 1,1, output_mode=mode)(img,filters)
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

    def c_code_cleanup(self, node, name, input_names, output_names, sub):
        """
        TODO: implement from c_code()???
        """
        return ""

    def c_support_code(self):
        return """
#define STRIDES(arr) ((arr)->strides)
#define FULL  2
#define SAME  1
#define VALID 0
#include <iostream>
using namespace std;
"""
    def c_code(self, node, name, (img2d, filtersflipped), (z, ), sub):
        if node.inputs[0].type != node.inputs[1].type:
            raise NotImplementedError()
        code="""
int mode=-1,typenum;
PyArrayObject *ain1=NULL, *ain2=NULL, *aout=NULL;
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
PyObject *img2d, *contig, *filtersflipped;
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
if (!PyArray_ISCONTIGUOUS(img2d)){
    contig = (PyObject*)(PyArray_GETCONTIGUOUS((PyArrayObject*)img2d));
    Py_DECREF(img2d);
    img2d = contig;
}
if (!PyArray_ISCONTIGUOUS(img2d)){
    PyErr_SetString(PyExc_ValueError, "img2d isn't contiguous");
    %(fail)s;
}

filtersflipped = PyArray_Newshape(%(filtersflipped)s,&kerns_shape, PyArray_CORDER);
if (!PyArray_ISCONTIGUOUS(filtersflipped)){
    contig = (PyObject*)(PyArray_GETCONTIGUOUS((PyArrayObject*)filtersflipped));
    Py_DECREF(filtersflipped);
    filtersflipped = contig;
}
if (!PyArray_ISCONTIGUOUS(filtersflipped)){
    PyErr_SetString(PyExc_ValueError, "filtersflipped isn't contiguous");
    %(fail)s;
}

if(s=="valid") mode=0;
else if(s=="full") mode=2;
else {PyErr_SetString(PyExc_ValueError, "invalid mode, only full and valid are supported"); %(fail)s;};
typenum = PyArray_ObjectType((PyObject*)%(img2d)s, 0);
typenum = PyArray_ObjectType((PyObject*)%(filtersflipped)s, 0);
if (typenum < 0) {PyErr_SetString(PyExc_ValueError, "Invalid type"); %(fail)s;}

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
  npy_intp dims[4] = {0,0,0,0}; //(npy_intp *)malloc(4*sizeof(%(type)s));
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

    aout = (PyArrayObject *)PyArray_SimpleNewFromData(2,dim_zz,
      typenum,PyArray_GETPTR2(%(z)s,b,n_kern));
    if (aout == NULL) %(fail)s;
    %(type)s *out=(%(type)s *)(aout->data);
    for (int i = 0; i < dim_zz[0]*dim_zz[1]; ++i) out[i] = 0;

    for(int stack_size=0;stack_size<%(self_imshp0)s;stack_size++){
      ain1 = (PyArrayObject *)PyArray_SimpleNewFromData(2,dim_im,
               type_im,PyArray_GETPTR2(img2d,b,stack_size));
      ain2 = (PyArrayObject *)PyArray_SimpleNewFromData(2,dim_ker,
               type_ker,PyArray_GETPTR2(filtersflipped,n_kern,stack_size));

      if (ain1 == NULL) %(fail)s;
      if (ain2 == NULL) %(fail)s;
      if (dim_im[0] != ((PyArrayObject*)img2d)->dimensions[2]) %(fail)s;
      if (dim_im[1] != ((PyArrayObject*)img2d)->dimensions[3]) %(fail)s;
      if (dim_ker[0] !=((PyArrayObject*)filtersflipped)->dimensions[2]) %(fail)s;
      if (dim_ker[1] !=((PyArrayObject*)filtersflipped)->dimensions[3]) %(fail)s;
      if (ain1->strides[0] != ain1->dimensions[1] * sizeof(%(type)s)) %(fail)s;
      if (ain2->strides[0] != ain2->dimensions[1] * sizeof(%(type)s)) %(fail)s;
      if (aout->strides[0] != aout->dimensions[1] * sizeof(%(type)s)) %(fail)s;
      if (ain1->strides[1] != sizeof(%(type)s)) %(fail)s;
      if (ain2->strides[1] != sizeof(%(type)s)) %(fail)s;
      if (aout->strides[1] != sizeof(%(type)s)) %(fail)s;

      %(type)s *in=(%(type)s *)(ain1->data);
      %(type)s *hvals=(%(type)s *)(ain2->data);

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
              %(type)s * idx2=&hvals[j*dim_ker[1]];
              if(ind0 < 0 || ind0 >= dim_im[0]){
                if(fill_value!=0)
                  for (int k=0; k < dim_ker[1]; k++) {
                    sum+= idx2[k] * fill_value;
                  }
              }else{
                //do the part where kernel is to the right of the img

                int k=0,max_k=max((int)(n-dim_im[1])+1,0);
                if(fill_value!=0){ 
                
                  for(k=0;k<max_k;k++){
                    sum+= idx2[k]*fill_value;
                  }
                }else {k=max_k;}
                
                //do the part where the kernel is on the img
                max_k=min(n+1,(int)dim_ker[1]);
                %(type)s * idx1=&in[ind0*dim_im[1]];
                for (int ind1=n-k; k<max_k; k++,ind1--) {
                  sum+= idx2[k] * idx1[ind1];
                }
                //do the part to the left of the img
                if(fill_value!=0)
                  for(;k<dim_ker[1];k++) sum+= idx2[k]*fill_value;
              }
            }else{
              %(type)s* idx1=&in[ind0*dim_im[1]]; //JB: should be dim_im[1] right? (was dim_im[0])
              %(type)s* idx2=&hvals[j*dim_ker[1]];
              int new_n = (n+dim_ker[1]-1);

              for (int k=0,last=new_n; k < dim_ker[1]; k++,last--) {
                sum+=idx2[k]*idx1[last];
              }
            }
          }//for j
          out[m*dim_zz[1]+n] %(affectation)s sum;
        }//for n
      }//for m
      Py_DECREF(ain1);
      Py_DECREF(ain2);
    }//for stack_size
    if (0 && (mode==FULL)){
      for (int i = 0; i < dim_zz[0]*dim_zz[1]; ++i) 
        std::cout << " " << out[i];
      std::cout << "\\n";
    }
    Py_DECREF(aout);
  }//for n_kern
}//for b
Py_XDECREF(img2d);
Py_XDECREF(filtersflipped);

fail:
        """
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
        if self.dtype=="float32": d["type"]="float"
        elif self.dtype=="float64": d["type"]="double"
        else: raise Exception("Type %s not implemented"%self.dtype)
        return code % d


def convolve2(kerns, kshp, nkern, images, imshp, bsize, step=(1,1),
              bias=None, mode='valid'):

    # if imshp, is a tuple, images contains one input dimension
    nvis_dim = 1 if len(imshp)!=3 else imshp[0]

    # all these reshapes should happen in place
    imrshp   = tensor.as_tensor([bsize] + list(imshp))
    imtensor = tensor.reshape(images, imrshp)

    kernrshp   = tensor.as_tensor([nkern, nvis_dim] + list(kshp))
    kerntensor = tensor.reshape(kerns, kernrshp)
   
    print '***** convolve2 *****'
    print 'imrshp = ', imrshp
    print 'kernrshp = ', kernrshp

    convop = ConvOp(imshp, kshp, nkern, bsize, 1, 1, output_mode=mode)
    convout = convop(imtensor, kerntensor)
   
    if bias:
        biastensor = tensor.DimShuffle((False,), ('x',0,'x','x'), inplace=True)(bias)
        convout = convout + biastensor
        
    rval = tensor.flatten(convout, 2)
    return rval, N.hstack((nkern, convop.outshp))

