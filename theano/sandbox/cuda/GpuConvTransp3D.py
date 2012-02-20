import numpy

import theano.tensor as T
from theano.misc import strutil
import theano

from theano.tensor.nnet.ConvTransp3D import ConvTransp3D
from theano.gof import local_optimizer

from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable
from theano.sandbox.cuda.opt import register_opt
from theano.sandbox.cuda import (CudaNdarrayType, HostFromGpu,
                                 host_from_gpu, GpuOp)


class GpuConvTransp3D(GpuOp):
    """ The gpu version of ConvTransp3D """
    def __eq__(self,other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, W, b, d, H, RShape = None):
        W_ = as_cuda_ndarray_variable(W)
        b_ = as_cuda_ndarray_variable(b)
        d_ = T.as_tensor_variable(d)
        H_ = as_cuda_ndarray_variable(H)
        if RShape:
            RShape_ = T.as_tensor_variable(RShape)
        else:
            RShape_ = T.as_tensor_variable([-1,-1,-1])

        return theano.Apply(self, inputs=[W_,b_,d_,H_, RShape_],
                            outputs = [CudaNdarrayType(dtype=H_.dtype,
                                                       broadcastable=(False,)*5)()])

    def infer_shape(self, node, input_shapes):
        W,b,d,H,RShape = node.inputs
        W_shape, b_shape, d_shape, H_shape, RShape_shape = input_shapes
        return [(H_shape[0], W_shape[1], RShape[0], RShape[1], RShape[2])]


    def perform_(self, node, inputs, output_storage):
        W, b, d, H, RShape = inputs
        print "\t\t\t\tGpuConvTransp3D python code still uses old format"
        output_storage[0][0] = computeR(W,b,d,H,RShape)

    def c_code_cache_version(self):
        return ()

    def c_code(self, node, nodename, inputs, outputs, sub):
        W, b, d, H, RShape = inputs
        fail = sub['fail']

        R = outputs[0]

        codeSource =  """
            ///////////// < code generated by GpuConvTransp3D >

            //printf("\t\t\t\tGpuConvTransp c code\\n");

            //Check dimensionality of inputs
            if (%(H)s->nd != 5)
            {
                PyErr_Format(PyExc_ValueError, "GpuConvTransp3D: H must be a 5-D tensor but it is %%i-D",%(H)s->nd);
                %(fail)s
            }

            if (%(W)s->nd != 5)
            {
                PyErr_Format(PyExc_ValueError, "GpuConvTransp3D: W must be a 5-D tensor");
                %(fail)s
            }

            if (%(b)s->nd != 1)
            {
                PyErr_Format(PyExc_ValueError, "GpuConvTransp3D: b must be a vector");
                %(fail)s
            }

            if (%(d)s->nd != 1)
            {
                PyErr_Format(PyExc_ValueError, "GpuConvTransp3D: d must be a vector");
                %(fail)s
            }

            //Read and check stride arguments
            if (%(d)s->dimensions[0] != 3)
            {
                PyErr_Format(PyExc_ValueError,"GpuConvTransp3D: 3 stride length arguments expected (for row, col, and time) but %%li were given", %(d)s->dimensions[0]);
                %(fail)s
            }
{ // for fail
            const int dr = *(dtype_%(d)s*)PyArray_GETPTR1(%(d)s,0);
            const int dc = *(dtype_%(d)s*)PyArray_GETPTR1(%(d)s,1);
            const int dt = *(dtype_%(d)s*)PyArray_GETPTR1(%(d)s,2);
            if (dr <= 0 || dc <= 0 || dt <= 0)
            {
                PyErr_Format(PyExc_ValueError, "GpuConvTransp3D: Strides must all be positive but are %%i, %%i, %%i",dr,dc,dt);
                %(fail)s
            }


            //Read and check sizes of inputs

{ // for fail
            const int batchSize = CudaNdarray_HOST_DIMS(%(H)s)[0];
            const int outputChannels =  CudaNdarray_HOST_DIMS(%(W)s)[0];

            if (CudaNdarray_HOST_DIMS(%(H)s)[4] != outputChannels)
            {
                PyErr_Format(PyExc_ValueError, "W produces a %%i channel image but the image has %%i channels. W.shape: (%%i, %%i, %%i,%%i, %%i) H.shape: (%%i, %%i, %%i, %%i, %%i)",outputChannels,CudaNdarray_HOST_DIMS(%(H)s)[4], CudaNdarray_HOST_DIMS(%(W)s)[0], CudaNdarray_HOST_DIMS(%(W)s)[1], CudaNdarray_HOST_DIMS(%(W)s)[2], CudaNdarray_HOST_DIMS(%(W)s)[3], CudaNdarray_HOST_DIMS(%(W)s)[4], CudaNdarray_HOST_DIMS(%(H)s)[0], CudaNdarray_HOST_DIMS(%(H)s)[1], CudaNdarray_HOST_DIMS(%(H)s)[2], CudaNdarray_HOST_DIMS(%(H)s)[3], CudaNdarray_HOST_DIMS(%(H)s)[4]);
                %(fail)s
            }
{ // for fail

            const int inputChannels = CudaNdarray_HOST_DIMS(%(W)s)[4];

            if (CudaNdarray_HOST_DIMS(%(b)s)[0] != inputChannels)
            {
                PyErr_Format(PyExc_ValueError, "ConvTransp3D: b operates on a %%i channel image but the image has %%i channels", CudaNdarray_HOST_DIMS(%(b)s)[0], inputChannels );
                %(fail)s
            }
{ // for fail

            const int filterHeight = CudaNdarray_HOST_DIMS(%(W)s)[1];
            const int filterWidth = CudaNdarray_HOST_DIMS(%(W)s)[2];
            const int filterDur = CudaNdarray_HOST_DIMS(%(W)s)[3];
            const int outputHeight = CudaNdarray_HOST_DIMS(%(H)s)[1];
            const int outputWidth = CudaNdarray_HOST_DIMS(%(H)s)[2];
            const int outputDur = CudaNdarray_HOST_DIMS(%(H)s)[3];

            int videoHeight = (outputHeight-1) * dr + filterHeight;
            int videoWidth = (outputWidth-1) * dc + filterWidth;
            int videoDur = (outputDur-1) * dt + filterDur;


            if (%(RShape)s)
            {
                if (%(RShape)s->nd != 1)
                {
                    PyErr_Format(PyExc_ValueError, "RShape must be a vector");
                    %(fail)s
                }

                if (%(RShape)s->dimensions[0] != 3)
                {
                    PyErr_Format(PyExc_ValueError, "RShape must specify a 3D shape ( [height,width,duration] )");
                    %(fail)s
                }
{ // for fail

                                dtype_%(RShape)s RShape0 = *(dtype_%(RShape)s*)PyArray_GETPTR1(%(RShape)s,0);
                                dtype_%(RShape)s RShape1 = *(dtype_%(RShape)s*)PyArray_GETPTR1(%(RShape)s,1);
                                dtype_%(RShape)s RShape2 = *(dtype_%(RShape)s*)PyArray_GETPTR1(%(RShape)s,2);

                if (RShape0 != -1)
                {
                    if (RShape0 < videoHeight || RShape1 < videoWidth || RShape2 < videoDur)
                    {
                        PyErr_Format(PyExc_ValueError, "Reconstruction must have shape of at least [%%i,%%i,%%i] but RShape argument requests that it be [%%i,%%i,%%i]" , videoHeight, videoWidth, videoDur, RShape0, RShape 1, RShape2 );
                        %(fail)s
                    }

                    videoHeight = RShape0;
                    videoWidth = RShape1;
                    videoDur = RShape2;
                }
            }

            //Allocate the reconstruction
            npy_intp dims[5];
            dims[0] = batchSize;
            dims[4] = inputChannels;
            dims[1] = videoHeight;
            dims[2] = videoWidth;
            dims[3] = videoDur;

                        if(!(%(R)s) || CudaNdarray_HOST_DIMS(%(R)s)[0]!=dims[0] ||
                        CudaNdarray_HOST_DIMS(%(R)s)[1]!=dims[1] ||
                        CudaNdarray_HOST_DIMS(%(R)s)[2]!=dims[2] ||
                        CudaNdarray_HOST_DIMS(%(R)s)[3]!=dims[3] ||
                        CudaNdarray_HOST_DIMS(%(R)s)[4]!=dims[4]){
                    Py_XDECREF(%(R)s);
               %(R)s = (CudaNdarray*)CudaNdarray_NewDims(5,dims);
                if (!(%(R)s)) {
                    PyErr_Format(PyExc_MemoryError,"Could not allocate R");
                    %(fail)s;
                }
                        }
            cudaMemset(%(R)s->devdata, 0, 4 * batchSize * inputChannels * videoHeight * videoWidth * videoDur);

{ // for fail

bool out_contiguous = CudaNdarray_is_c_contiguous(%(R)s);
int version = -1;
int verbose = 0;
bool subsample =(dr>1)||(dc>1)||(dt>1);
bool b_strided = (CudaNdarray_HOST_STRIDES(%(b)s)[0]!=1) && !(CudaNdarray_HOST_STRIDES(%(b)s)[0]==0 && outputChannels==1);
printf("b stride0=%%d\\n",CudaNdarray_HOST_STRIDES(%(b)s)[0]);
bool work_complete = false;

const int ws4 = CudaNdarray_HOST_STRIDES(%(W)s)[4];
const int ws3 = CudaNdarray_HOST_STRIDES(%(W)s)[3];
const int ws2 = CudaNdarray_HOST_STRIDES(%(W)s)[2];
const int ws1 = CudaNdarray_HOST_STRIDES(%(W)s)[1];
const int ws0 = CudaNdarray_HOST_STRIDES(%(W)s)[0];
const int hs4 = CudaNdarray_HOST_STRIDES(%(H)s)[4];
const int hs3 = CudaNdarray_HOST_STRIDES(%(H)s)[3];
const int hs2 = CudaNdarray_HOST_STRIDES(%(H)s)[2];
const int hs1 = CudaNdarray_HOST_STRIDES(%(H)s)[1];
const int hs0 = CudaNdarray_HOST_STRIDES(%(H)s)[0];


if(out_contiguous && (version==0||version==-1) && outputDur<=512 && !work_complete){
    //conv_transp_rows_stack
    dim3 grid(batchSize * inputChannels, videoHeight * videoWidth);
    dim3 threads(videoDur);

HERE

    int shared_size=0;
        conv_transp_rows_stack<<<grid, threads, shared_size>>>(
        CudaNdarray_DEV_DATA(%(H)s), CudaNdarray_DEV_DATA(%(W)s), CudaNdarray_DEV_DATA(%(b)s), CudaNdarray_DEV_DATA(%(R)s),
        videoHeight, videoWidth, videoDur,
        filterHeight, filterWidth, filterDur,
        outputHeight, outputWidth, outputDur,
        outputChannels, inputChannels,
        dr,dc,dt,
        hs3,hs2,hs1,hs4,hs0,
        ws3,ws2,ws1,ws4,ws0,
        CudaNdarray_HOST_STRIDES(%(b)s)[0]);

        CNDA_THREAD_SYNC;
        cudaError_t sts = cudaGetLastError();
        if (cudaSuccess == sts)
        {
            work_complete = true;
        if (verbose>1) printf("threads.x=%%i, threads.y=%%i, grid.x=%%i, grid.y=%%i, shared_size=%%i, nb_threads=%%i\\n", threads.x, threads.y, grid.x, grid.y, shared_size, threads.x * threads.y);
            if (verbose) printf("INFO: used 'conv_transp_rows_stack' version\\n");
        }
        else
        {
            if (verbose) printf("threads.x=%%i, threads.y=%%i, grid.x=%%i, grid.y=%%i, shared_size=%%i, nb_threads=%%i\\n", threads.x, threads.y, grid.x, grid.y, shared_size, threads.x * threads.y);
            if (verbose) printf("ERROR: all implementations failed for GpuConvTransp3D! (%%s)",cudaGetErrorString(sts));
            PyErr_Format(PyExc_RuntimeError, "ERROR: all implementations failed for GpuConvTransp3D! (%%s)",
                    cudaGetErrorString(sts));
            %(fail)s
        }


}

if(!work_complete){
            PyErr_Format(PyExc_RuntimeError, "ERROR: no implementations executed for this GpuConvTransp3D! out_contiguous=%%d b_strided=%%d outputDur=%%d",
                         out_contiguous,b_strided,outputDur);
            %(fail)s
}



}}}}}} // for fail
            ///////////// < /code generated by GpuConvTransp3D >
        """
        return strutil.renderString(codeSource,locals())

    def c_support_code_apply(self, node, nodename):
        # This code is not sensitive to the ignore_border flag.
        # It runs for every position in the output z, and then computes the gradient for the
        # input pixels that were downsampled to that z-position.
        codeSource =  """
__global__ void
//thread block size = videoDur
//grid block size =(batchSize * inputChannels, videoHeight * videoWidth)
//


conv_transp_rows_stack( float* H, float* kern, float* bias, float* R,
         int img_len, int img_wid, int img_dur,
                 int kern_len, int kern_wid, int kern_dur,
                 int H_len, int H_wid, int H_dur,
         int nkern, int nstack,
                 int dr, int dc, int dt,
                 int H_stride_frame, int H_stride_col, int H_stride_row,
         int H_stride_stack, int H_stride_batch,
                 int kern_stride_frame, int kern_stride_col, int kern_stride_row,
         int kern_stride_stack, int kern_stride_nkern,
                 int bias_stride)
{
    int __shared__ batch_id, stack_id;
    float  __shared__ *d_img, *d_kern;

    batch_id= blockIdx.x/nstack;
    stack_id = blockIdx.x - batch_id*nstack;

    const int R_row = blockIdx.y/img_wid;
    const int R_col = blockIdx.y%img_wid;
    const int R_frame=threadIdx.x;

    const int r = R_row;
    const int c = R_col;
    const int t = R_frame;

    const int ftc = max(0, int(ceil(float(t-kern_dur +1  )/float(dt))));
    const int fcc = max(0, int(ceil(float(c-kern_wid +1)/float(dc))));
    int rc =  max(0, int(ceil(float(r-kern_len+1)/float(dr))));

    float sum = 0;
    while(rc < H_len){
    int rk = r - rc * dr;
        if(rk < 0)
            break;
        int cc = fcc;
        while( cc < H_wid){
            int ck = c - cc * dc;
            if(ck < 0)
                break;
            int tc = ftc;
            while(tc < H_dur){
                int tk = t - tc * dt;
                if(tk < 0)
                    break;
                //R[i,j,r,c,t] += numpy.dot(W[:,j,rk,ck,tk], H[i,:,rc,cc,tc] )
                        for(int q=0;q<nkern;q++){
                          sum += kern[q*kern_stride_nkern+stack_id*kern_stride_stack+rk*kern_stride_row+ck*kern_stride_col+tk*kern_stride_frame]*
                                 H[batch_id*H_stride_batch+q*H_stride_stack+rc*H_stride_row+cc*H_stride_col+tc*H_stride_frame];
                        }

                tc += 1;
                }
            cc += 1;
        }
        rc += 1;
    }
    R[batch_id*nstack*img_len*img_wid*img_dur+//the good batch
      stack_id+//the output image
      R_row*img_wid*img_dur*nstack+//the output row
      R_col*img_dur*nstack + //the output_col
      R_frame*nstack] = sum + bias[stack_id*bias_stride];

}

"""
        return codeSource


gpu_conv_transpd = GpuConvTransp3D()

@register_opt()
@local_optimizer([])
def local_gpu_conv_transpd(node):
    if isinstance(node.op, ConvTransp3D):
        if numpy.any([i.owner and isinstance(i.owner.op, HostFromGpu) for i in node.inputs]):
            if numpy.all([o.type.dtype == 'float32' for o in node.outputs]):
                W, b, d, H, RShape = node.inputs
                return [host_from_gpu(gpu_conv_transpd(W, b, d, H, RShape))]


#If the input size wasn't a multiple of D we may need to cause some automatic padding to get the right size of reconstruction
def computeR(W,b,d,H,Rshape = None):
        assert len(W.shape) == 5
        assert len(H.shape) == 5
        assert len(b.shape) == 1
        assert len(d) == 3


        outputChannels, inputChannels, filterHeight, filterWidth, filterDur = W.shape
        batchSize, outputChannelsAgain, outputHeight, outputWidth, outputDur = H.shape
        assert outputChannelsAgain == outputChannels
        assert b.shape[0] == inputChannels

        dr,dc,dt = d
        assert dr > 0
        assert dc > 0
        assert dt > 0

        videoHeight = (outputHeight-1) * dr + filterHeight
        videoWidth = (outputWidth-1) * dc + filterWidth
        videoDur = (outputDur-1) * dt + filterDur

        if Rshape != None and Rshape[0] != -1:
            if Rshape[0] < videoHeight:
                print (Rshape[0], videoHeight)
                assert False
            assert Rshape[1] >= videoWidth
            assert Rshape[2] >= videoDur

            #print "setting video size to Rshape = "+str(Rshape)

            videoHeight, videoWidth, videoDur = Rshape
        #else:
        #    print "No Rshape passed in"

        #print "video size: "+str((videoHeight, videoWidth, videoDur))

        R =  numpy.zeros( (batchSize, inputChannels, videoHeight,
            videoWidth, videoDur ) , dtype=H.dtype)

        #R[i,j,r,c,t] = b_j + sum_{rc,rk | d \circ rc + rk = r} sum_{cc,ck | ...} sum_{tc,tk | ...} sum_k W[k, j, rk, ck, tk] * H[i,k,rc,cc,tc]
        for i in xrange(0,batchSize):
            #print '\texample '+str(i+1)+'/'+str(batchSize)
            for j in xrange(0,inputChannels):
                #print '\t\tfeature map '+str(j+1)+'/'+str(inputChannels)
                for r in xrange(0,videoHeight):
                    #print '\t\t\trow '+str(r+1)+'/'+str(videoHeight)
                    for c in xrange(0,videoWidth):
                        for t in xrange(0,videoDur):
                            R[i,j,r,c,t] = b[j]

                            ftc = max([0, int(numpy.ceil(float(t-filterDur +1  )/float(dt))) ])
                            fcc = max([0, int(numpy.ceil(float(c-filterWidth +1)/float(dc))) ])

                            rc =  max([0, int(numpy.ceil(float(r-filterHeight+1)/float(dr))) ])
                            while rc < outputHeight:
                                rk = r - rc * dr
                                if rk < 0:
                                    break

                                cc = fcc
                                while cc < outputWidth:
                                    ck = c - cc * dc
                                    if ck < 0:
                                        break

                                    tc = ftc
                                    while tc < outputDur:
                                        tk = t - tc * dt
                                        if tk < 0:
                                            break

                                        R[i,j,r,c,t] += numpy.dot(W[:,j,rk,ck,tk], H[i,:,rc,cc,tc] )

                                        tc += 1
                                    "" #close loop over tc
                                    cc += 1
                                "" #close loop over cc

                                rc += 1
                            "" #close loop over rc
                        "" #close loop over t
                    "" #close loop over c
                "" #close loop over r
            "" #close loop over j
        "" #close loop over i

        return R
