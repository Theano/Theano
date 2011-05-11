import theano
from theano.tensor import basic as T
import numpy as N
#from util import strutil
from theano import printing
from theano.tensor.blas_headers import blas_header_text
from theano.tensor.blas import ldflags
from theano.misc import strutil


#Note: not a true convolution because we don't bother with flipping the kernel
#An op that takes a weight tensor W. a bias vector b, and a visible tensor V, produces a hidden unit tensor H
#Also parmeterized by integer strides dr,dc,dt
#H[i,r,c,t,j] = video i within the minibatch, feature map j, location and time within feature map (r,c,t)
#W[j,k,l,m,z] = weights connecting H[i,r,c,t,j] to V[i,dr*r+k,dc*c+l,dt*t+m,z]
#b[j] = bias of feature map j
#V[i,r,c,t,j] = pixel at (r,c,t) within video featuremap j of video i within the minibatch
#i.e., H[i,j,r,c,t] = b_j + sum_k sum_l sum_m sum_z W[j,k,l,m,z] V[i,z, dr*r+k,dc*c+l,dt*t+m]
#The layouts of these variables are chosen to improve locality of reference.
#numpy seems to put the largest stride on axis 0 and decrease the stride from there. If we do convolution
#one filter at a time, one example at a time, then we want the largest strides to
#be over the examples. We want the smallest stride to be over the input channel because as we change
#the channel we re-visit the same location in the input.
#The smallest stride being over the input channel means that the weights need to be formatted with the input
#channel as the last index

#partial C / partial b_j =  sum_i sum_k sum_r sum_c sum_t (partial C / partial H[i,r,c,t,k] ) * ( partial H[i,r,c,t,k] / partial b_j )
# =  sum_i sum_k sum_r sum_c sum_t (partial C / partial H[i,r,c,t,k] )  * delta(k = j)
# =  sum_i sum_r sum_c sum_t (partial C / partial H[i,r,c,t,j] )


#partial C / partial W[j,k,l,m,z] = sum_i sum_n sum_p sum_q sum_r (partial C /partial H[i,p,q,r,n] ) * (partial H[i,p,q,r,n] / partial W[j,k,l,m,z])
# = partial C / partial W[j,k,l,m,z] = sum_i sum_n sum_p sum_q sum_r (partial C /partial H[i,p,q,r,n] ) *
# (partial sum_s sum_u sum_v sum_a  W[n,a, s,u,v] V[i, dr*p+s,dc*q+u,dt*r+v, a] ) / partial W[j,k,l,m,z])
# = partial C / partial W[j,k,l,m,z] = sum_i sum_p sum_q sum_r (partial C /partial H[i,p,q,r,j] ) *
# (partial sum_s sum_u sum_v sum_a W[j,a, s,u,v] V[i,dr*p+s,dc*q+u,dt*r+v,a] ) / partial W[j,k,l,m,z])
# = partial C / partial W[j,k,l,m,z] = sum_i sum_p sum_q sum_r (partial C /partial H[i,p,q,r,j] ) *  V[i,dr*p+k,dc*q+l,dt*r+m,z]

#derivatives wrt V unimplemented for now. derivatives wrt dr, dc, dt are undefined since dr, dc, dt are natural numbers.

class Conv3D(theano.Op):
    """ 3D "convolution" of multiple filters on a minibatch (does not flip the kernel, moves kernel with a user specified stride) """
    def __eq__(self,other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return "Conv3D"

    def c_code_cache_version(self):
        return (2,)


    def make_node(self, V, W, b, d):
        """
            :param V: Visible unit, input(batch,row,column,time,in channel)
            :param W: Weights, filter(out channel,row,column,time,in channel)
            :param b: bias, shape == (W.shape[0],)
            :param d: strides when moving the filter over the input(dx,dy,dt)
        """

        V_ = T.as_tensor_variable(V)
        W_ = T.as_tensor_variable(W)
        b_ = T.as_tensor_variable(b)
        d_ = T.as_tensor_variable(d)

        node = theano.Apply(self, inputs=[V_, W_,b_,d_], outputs = [ T.TensorType(V_.dtype, (V_.broadcastable[0],False,False,False, W_.broadcastable[0]))() ] )


        return node

    def grad(self,inputs, output_gradients):
        V,W,b,d = inputs
        dCdH ,= output_gradients
        #make all of these ops support broadcasting of scalar b to vector b and eplace the zeros_like in all their grads
        #print dCdH.broadcastable
        #print "dCdH.broadcastable"
        #quit(-1)
        #dCdH = printing.Print("dCdH = ",["shape"])

        dCdV = ConvTransp3D.convTransp3D(W, T.zeros_like(V[0,0,0,0,:]), d, dCdH, V.shape[1:4] )
        WShape = W.shape
        dCdW = ConvGrad3D.convGrad3D(V,d,WShape,dCdH)
        dCdb = T.sum(dCdH, axis=(0,1,2,3))
        dCdd = None #not differentiable, since d is not continuous

        if 'name' in dir(dCdH) and dCdH.name is not None:
            dCdH_name = dCdH.name
        else:
            dCdH_name = 'anon'

        if 'name' in dir(V) and V.name is not None:
            V_name = V.name
        else:
            V_name = 'anon'

        if 'name' in dir(W) and W.name is not None:
            W_name = W.name
        else:
            W_name = 'anon'

        if 'name' in dir(b) and b.name is not None:
            b_name = b.name
        else:
            b_name = 'anon'

        dCdV.name = 'Conv3D_dCdV.dCdH='+dCdH_name+',V='+V_name
        dCdW.name = 'Conv3D_dCdW.dCdH='+dCdH_name+',V='+V_name+',W='+W_name
        dCdb.name = 'Conv3D_dCdb.dCdH='+dCdH_name+',V='+V_name+',W='+W_name+',b='+b_name



        return [ dCdV, dCdW, dCdb, dCdd ]

    def perform(self, node, inputs, output_storage):
        V, W, b, d = inputs
        print "Conv3D python code"
        output_storage[0][0] = computeH(V,W,b,d)

    def infer_shape(self, node, input_shapes):
        V,W,b,d = node.inputs
        V_shape, W_shape, b_shape, d_shape = input_shapes

        dr = d[0]
        dc = d[1]
        dt = d[2]
        batch_size = V_shape[0]
        output_channels = W_shape[0]
        vidHeight = V_shape[1]
        filterHeight = W_shape[1]
        vidWidth = V_shape[2]
        filterWidth = W_shape[2]
        vidDur = V_shape[3]
        filterDur = W_shape[3]

        output_height = T.floor((vidHeight - filterHeight) // dr) + 1
        output_width = T.floor((vidWidth - filterWidth) // dc) + 1
        output_dur = T.floor((vidDur - filterDur) // dt) + 1

        rval = (batch_size,  output_height, output_width, output_dur, output_channels )


        return [ rval ]

    def c_support_code(self):
        return blas_header_text()

    def c_libraries(self):
        return ldflags()

    def c_compile_args(self):
        flags =  ldflags(libs=False, flags=True)
        flags.append('-Werror')
        return flags

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_header_dirs(self):
        return ldflags(libs=False, include_dir=True)

    def c_code(self, node, nodename, inputs, outputs, sub):
        V, W, b, d = inputs
        fail = sub['fail']

        H = outputs[0]


        codeSource =  """
            ///////////// < code generated by Conv3D >

            //printf("\t\t\t\tConv3D c code\\n");

            //Check dimensionality of inputs
            if (%(W)s->nd != 5)
            {
                PyErr_Format(PyExc_ValueError, "Conv3D: W must be a 5 dimensional tensor");
                            %(fail)s

            }

            if (%(V)s->nd != 5)
            {
                PyErr_Format(PyExc_ValueError, "Conv3D: V must be a 5 dimensional tensor");
                            %(fail)s
            }

            if (%(b)s->nd != 1)
            {
                PyErr_Format(PyExc_ValueError,"Conv3D: b must be a vector.");
                %(fail)s
            }

            if (%(d)s->nd != 1)
            {
                PyErr_Format(PyExc_ValueError,"Conv3D: d must be a vector.");
                %(fail)s
            }

            if (%(d)s->dimensions[0] != 3)
            {
                PyErr_Format(PyExc_ValueError,"Conv3D: 3 stride length arguments expected (row, col, time) but %%li were given", (long)%(d)s->dimensions[0]);
                %(fail)s
            }

            //Read and check sizes of inputs
{ // exta scope so error handler jumps don't cause errors
            const int batchSize = %(V)s->dimensions[0];
            const int outputChannels =  %(W)s->dimensions[0];
            const int inputChannels = %(V)s->dimensions[4];

            if (%(W)s->dimensions[4] != inputChannels)
            {
                PyErr_Format(PyExc_ValueError, "Conv3D: W operates on a %%ld channel image but the image has %%d channels. Overall shape of input: (%%ld,%%ld,%%ld,%%ld,%%ld)", (long)%(W)s->dimensions[4], inputChannels, (long)%(V)s->dimensions[0], (long)%(V)s->dimensions[1], (long)%(V)s->dimensions[2], (long)%(V)s->dimensions[3], (long)%(V)s->dimensions[4]);
                %(fail)s
            }

            if (%(b)s->dimensions[0] != outputChannels)
            {
                PyErr_Format(PyExc_ValueError, "Conv3D: b adds to a(n) %%ld channel output image but the output has %%d channels", (long)%(b)s->dimensions[0], outputChannels);
                %(fail)s
            }

{  //extra scope so error handler jumps don't cause errors
            const int filterHeight = %(W)s->dimensions[1];
            const int filterWidth = %(W)s->dimensions[2];
            const int filterDur = %(W)s->dimensions[3];
            const int vidHeight = %(V)s->dimensions[1];
            const int vidWidth = %(V)s->dimensions[2];
            const int vidDur = %(V)s->dimensions[3];\

            if (vidHeight < filterHeight)
            {
                PyErr_Format(PyExc_ValueError, "W has a height of %%i but V is only %%i pixels tall",filterHeight,vidHeight);
                %(fail)s
            }

{ // extra scope so fail works

            if (vidWidth < filterWidth)
            {
                PyErr_Format(PyExc_ValueError, "W has a width of %%i but V is only %%i pixels wide",filterWidth,vidWidth);
                %(fail)s
            }

{ // extra scope so fail works

            if (vidDur < filterDur)
            {
                PyErr_Format(PyExc_ValueError, "W has a duration of %%i but V is only %%i pixels long",filterDur,vidDur);
                %(fail)s
            }

{ // extra scope so fail works

            //Read and check stride arguments
            const int dr = *(dtype_%(d)s*) PyArray_GETPTR1(%(d)s,0);
            const int dc = *(dtype_%(d)s*) PyArray_GETPTR1(%(d)s,1);
            const int dt = *(dtype_%(d)s*) PyArray_GETPTR1(%(d)s,2);

            if (dr <= 0 || dc <= 0 || dt <= 0)
            {
                PyErr_Format(PyExc_ValueError,"Conv3D: Strides must all be positive but are %%i, %%i, %%i",dr,dc,dt);
                %(fail)s
            }
{ // extra scope so fail works

            //Make correctly sized output
            const long long outputHeight = int( (vidHeight - filterHeight) / dr )+1;
            const long long outputWidth = int( (vidWidth - filterWidth) / dc )+1;
            const long long outputDur = int( (vidDur - filterDur) / dt ) +1;


            npy_intp dims[5];
            dims[0] = batchSize;
            dims[4] = outputChannels;
            dims[1] = outputHeight;
            dims[2] = outputWidth;
            dims[3] = outputDur;



            if(!(%(H)s) || %(H)s->dimensions[0]!=dims[0] ||
            %(H)s->dimensions[1]!=dims[1] ||
            %(H)s->dimensions[2]!=dims[2] ||
            %(H)s->dimensions[3]!=dims[3] ||
            %(H)s->dimensions[4]!=dims[4]){
                Py_XDECREF(%(H)s);
                %(H)s = (PyArrayObject *) PyArray_SimpleNew(5, dims, %(V)s->descr->type_num);
                if (!(%(H)s)) {
                    PyErr_Format(PyExc_MemoryError,"Conv3D: Could not allocate output.");
                    %(fail)s
                }
            }
{ // extra scope so fail works


            #define ELEM_AT(x, i) * ( dtype_ ## x *) ( x->data + (i) )


            const int ws0 = %(W)s->strides[0];
            const int ws1 = %(W)s->strides[1];
            const int ws2 = %(W)s->strides[2];
            const int vs1 = %(V)s->strides[1];
            const int ws4 = %(W)s->strides[4];
            const int vs4 = %(V)s->strides[4];
            const int ws3 = %(W)s->strides[3];
            const int vs3 = %(V)s->strides[3];
            const int vs2 = %(V)s->strides[2];
            const int bs  = %(b)s->strides[0];
            const int hs4 = %(H)s->strides[4];



            // Compute H
            //H[i,j,x,y,t] = b_j + sum_k sum_l sum_m sum_z W[j,z,k,l,m] V[i,z, dr*r+k,dc*c+l,dt*t+m]
            //TODO: add special cases
            // ex: filterDur == 1 && batchSize == 1 && dt = 1  (for SFA)
            // ex: inputChannels == 1 """








        #if the data types are not mixed, we can insert special case optimizations based on BLAS
        VV, WV, bv, dv = node.inputs
        HV = node.outputs[0]
        if VV.dtype == WV.dtype and HV.dtype == VV.dtype:
            if VV.dtype == 'float64':
                gemv = 'dgemv_'
            elif VV.dtype == 'float32':
                gemv = 'sgemv_'
            else:
                raise Exception('Unrecognized dtype for convolution '+V.value.dtype)

            codeSource += """
            if (inputChannels > 20 && outputChannels > 20 && ws4 == sizeof(ELEM_AT(%(W)s,0)))
            {
              //std::cout << "lots of channels special case code" << std::endl;
              #define blas_type dtype_ ## %(V)s
              const blas_type  constant_one = 1.0;
              char N = 'T';
              int ws0e = ws0 / sizeof(ELEM_AT(%(W)s,0));
              int vs4e = vs4 / sizeof(ELEM_AT(%(V)s,4));
              int hs4e = hs4 / sizeof(ELEM_AT(%(H)s,4));

                //special case code for the "lots of channels" case
                //uses a BLAS matrix vector multiply to compute the contribute for
                //all channels of an input pixel to all channels of an output pixel
                //simultaneously
              long long Hpos = 0;
              long long Vpos = 0;
              for (int i = 0; i < batchSize; i++) {
                    long long Hposi = Hpos;
                    long long Vposi = Vpos;


                    for (int r = 0;  r < outputHeight; r++) {
                      long long Hposr = Hpos;
                      long long Vposr = Vpos;
                      for (int c = 0; c < outputWidth; c++) {
                       long long Hposc = Hpos;
                       long long Vposc = Vpos;
                       for (int t = 0; t < outputDur; t++) {
                            long long Hpost = Hpos;
                            long long Vpost = Vpos;
                            //of the loops so far, j should be the innermost, because
                            //each loop through j visits the same elements of V
                            //this implies that the last index of H should be the j index
                            //since V and H should have the same format, this means
                            //z should be the last index in v, and therefore the innermost
                            //of the next set of for loops

                            int Wpos = 0;
                            int bPos = 0;


                            long long Hposj = Hpos;
                            for (int j = 0; j < outputChannels; j++) {
                                // H[i,r,c,t,j] = b[j]
                                ELEM_AT(%(H)s,Hposj) = ELEM_AT(%(b)s,bPos);
                                Hposj += hs4;
                                bPos += bs;
                            }

                            dtype_%(H)s * writePos = & ELEM_AT(%(H)s,Hpos);


                            for (int k =0; k < filterHeight; k++) {
                                  int Wposk = Wpos;
                                  long long Vposk = Vpos;
                                  for (int l = 0; l < filterWidth; l++) {
                                    int Wposl = Wpos;
                                    long long Vposl = Vpos;
                                    for (int m = 0; m < filterDur; m++) {

                                      //H[i,r,c,t,:] += N.dot(W[:,k,l,m,:],V[i,dr*r+k,dc*c+l,dt*t+m,:])


                                      //note: changing the weights so that outputChannels and inputChannels were the last two rather than
                                      //the first and last elements did not speed this up, even for extremely large input sizes

                                      %(gemv)s(&N, & inputChannels, & outputChannels,
                     &constant_one, & ELEM_AT( %(W)s , Wpos),& ws0e,
                     & ELEM_AT(%(V)s, Vpos),& vs4e, &constant_one,
                     writePos,& hs4e);

                                      Wpos  += ws3;
                                      Vpos  += vs3;
                                    } // close m
                                    Wpos = Wposl + ws2;
                                    Vpos = Vposl + vs2;
                                  } //close l
                                  Wpos = Wposk + %(W)s->strides[1];
                                  Vpos = Vposk + %(V)s->strides[1];
                                } //close k
                             Hpos = Hpost + %(H)s->strides[3];
                             Vpos = Vpost + vs3 * dt;
                         } //close t
                         Hpos = Hposc + %(H)s->strides[2];
                         Vpos = Vposc + vs2 * dc;
                       } //close c
                       Hpos = Hposr + %(H)s->strides[1];
                       Vpos = Vposr + %(V)s->strides[1] * dr;
                   } //closes r
                   Hpos = Hposi + %(H)s->strides[0];
                   Vpos = Vposi + %(V)s->strides[0];
              } //closes i


            } //closes "lots of channels" special case code
            else
"""

        codeSource += """
            {
              //General case code
              //std::cout << "general case code" << std::endl;
              long long Hpos = 0;
              long long Vpos = 0;
              for (int i = 0; i < batchSize; i++) {
                    long long Hposi = Hpos;
                    long long Vposi = Vpos;


                    for (int r = 0;  r < outputHeight; r++) {
                      long long Hposr = Hpos;
                      long long Vposr = Vpos;
                      for (int c = 0; c < outputWidth; c++) {
                       long long Hposc = Hpos;
                       long long Vposc = Vpos;
                       for (int t = 0; t < outputDur; t++) {
                            long long Hpost = Hpos;
                            long long Vpost = Vpos;
                            //of the loops so far, j should be the innermost, because
                            //each loop through j visits the same elements of V
                            //this implies that the last index of H should be the j index
                            //since V and H should have the same format, this means
                            //z should be the last index in v, and therefore the innermost
                            //of the next set of for loops

                            int Wpos = 0;
                            int bPos = 0;


                            for (int j = 0; j < outputChannels; j++) {


                                long long Hposj = Hpos;
                                long long Vposj = Vpos;
                                int Wposj = Wpos;

                                // H[i,r,c,t,j] = b[j]

                                dtype_%(H)s & writePos = ELEM_AT(%(H)s,Hpos);


                                writePos = ELEM_AT(%(b)s,bPos);


                                for (int k =0; k < filterHeight; k++) {
                                  int Wposk = Wpos;
                                  long long Vposk = Vpos;
                                  for (int l = 0; l < filterWidth; l++) {
                                    int Wposl = Wpos;
                                    long long Vposl = Vpos;
                                    for (int m = 0; m < filterDur; m++) {
                                      int Wposm = Wpos;
                                      long long Vposm = Vpos;
                                      for (int z = 0; z < inputChannels; z++) {
                                        //H[i,r,c,t,j] += W[j,z,k,l,m] * V[i,dr*r+k, dc*c+l, dt*t+m,z]


                                        writePos += ELEM_AT(%(W)s,Wpos) * ELEM_AT(%(V)s,Vpos);

                                        Wpos += ws4;
                                        Vpos += vs4;
                                      } // close z
                                      Wpos = Wposm + ws3;
                                      Vpos = Vposm + vs3;
                                    } // close m
                                    Wpos = Wposl + ws2;
                                    Vpos = Vposl + vs2;
                                  } //close l
                                  Wpos = Wposk + %(W)s->strides[1];
                                  Vpos = Vposk + %(V)s->strides[1];
                                } //close k


                              bPos += bs;
                              Wpos = Wposj + ws0;
                              Hpos = Hposj +  hs4;
                              Vpos = Vposj;
                              //std::cout << "incremented Wpos by " << ws0 << std::endl;
                              //std::cout << "incremented Hpos by " << hs4 << std::endl;
                             } //close j
                             Hpos = Hpost + %(H)s->strides[3];
                             Vpos = Vpost + vs3 * dt;
                         } //close t
                         Hpos = Hposc + %(H)s->strides[2];
                         Vpos = Vposc + vs2 * dc;
                       } //close c
                       Hpos = Hposr + %(H)s->strides[1];
                       Vpos = Vposr + %(V)s->strides[1] * dr;
                   } //closes r
                   Hpos = Hposi + %(H)s->strides[0];
                   Vpos = Vposi + %(V)s->strides[0];
              } //closes i
            } //closes general case code
}}}}}}} //extra scope so error handler jumps don't cross declarations
            ///////////// < /code generated by Conv3D >
        """

        return strutil.renderString(codeSource,locals())

global conv3D
conv3D = Conv3D()

def computeH(V,W,b,d):
    assert len(W.shape) == 5
    assert len(V.shape) == 5
    if len(b.shape) != 1:
        print b.shape
        assert False
    assert len(d) == 3

    batchSize = V.shape[0]
    outputChannels = W.shape[0]
    inputChannels = V.shape[4]
    if W.shape[4] != inputChannels:
        raise Exception("W.shape[4] = "+str(W.shape[4])+" but inputChannels = "+str(inputChannels))
    filterHeight = W.shape[1]
    filterWidth = W.shape[2]
    filterDur = W.shape[3]
    vidHeight = V.shape[1]
    vidWidth = V.shape[2]
    vidDur = V.shape[3]
    assert vidHeight >= filterHeight
    assert vidWidth >= filterWidth
    assert vidDur >= filterDur
    dx,dy,dt = d
    assert dx > 0
    assert dy > 0
    assert dt > 0

    outputHeight = int( (vidHeight - filterHeight) / dx )+1
    outputWidth = int( (vidWidth - filterWidth) / dy )+1
    outputDur = int( (vidDur - filterDur) / dt ) +1


    H =  N.zeros( (batchSize,  outputHeight,
        outputWidth, outputDur, outputChannels ), dtype=V.dtype )

    #H[i,j,x,y,t] = b_j + sum_k sum_l sum_m sum_z W[j,z,k,l,m] V[i,z, dx*x+k,dy*y+l,dt*t+m]
    for i in xrange(0,H.shape[0]):
        #print '\texample '+str(i+1)+'/'+str(H.shape[0])
        for j in xrange(0,H.shape[4]):
                #print '\t\tfeature map '+str(j+1)+'/'+str(H.shape[1])
            for x in xrange(0,H.shape[1]):
                #print '\t\t\trow '+str(x+1)+'/'+str(H.shape[2])
                for y in xrange(0,H.shape[2]):
                    for t in xrange(0,H.shape[3]):
                        H[i,x,y,t,j] = b[j]
                        for k in xrange(0,filterHeight):
                            for l in xrange(0,filterWidth):
                                for m in xrange(0,filterDur):
                                    for z in xrange(0,inputChannels):
                                        #if (i,j,x,y,t) == (0,0,0,0,0):
                                        #    print (( W[j,z,k,l,m] , V[i,z,d[0]*x+k,d[1]*y+l,d[2]*t+m] ), (k,l,m) )
                                        w = W[j,k,l,m,z]
                                        v = V[i,d[0]*x+k, d[1]*y+l, d[2]*t+m,z]
                                        #if i == 0 and x == 0 and y == 0 and t == 0 and j == 0:
                                        #    print 'setting H[0] += '+str(w*v)+'   W['+str((j,z,k,l,m))+']='+str(w)+'   V['+str((i,d[0]*x+k,d[1]*y+l,d[2]*t+m,z))+']='+str(v)
                                        H[i,x,y,t,j] += w * v
    return H


import ConvGrad3D
import ConvTransp3D
