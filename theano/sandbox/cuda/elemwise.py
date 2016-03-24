"""
This file implement 3 different version of the elemwise op on the
gpu. Only NaiveAlgo is used and it is not very naive now.

The elemwise fct are also used with scalar operation! So it can happen
that ndim is 0 as with all scalar type.

"""
from __future__ import absolute_import, print_function, division


import logging

import numpy

from theano.scalar.basic import upgrade_to_float_no_complex, complex_types
from theano.scalar.basic_scipy import Erfinv
from six import StringIO
from six.moves import xrange
from theano import Apply
from theano import gof, scalar


_logger_name = 'theano.sandbox.cuda.elemwise'
_logger = logging.getLogger(_logger_name)


def _logical_scalar(x):
    return numpy.all(x.type.broadcastable)


def get_str_list_logical_scalar(node, value_str='ii_i%i_value',
                                data_str='ii_i%i_data[0]'):
    l = []
    for ipos, i in enumerate(node.inputs):
        if _logical_scalar(i):
            l += [value_str % ipos]
        else:
            l += [data_str % ipos]
    return l


class SupportCodeError(Exception):
    """
    It is currently not possible to auto-generate a GPU implementation for
    an elementwise Op with c_support_code_apply().
    But we support Op.c_support_code.

    """


class NaiveAlgo(object):
    """
    Parameters
    ----------
    scalar_op
        The scalar operation to execute on each element.
    sync
        If True, will wait after the kernel launch and check for error call.

    """

    verbose = 0  # 1, 2 or 3 for more verbose output.

    @property
    def cache_version(self):
        ver = self.scalar_op.c_code_cache_version()
        if ver:
            return (20, self.verbose, self.sync, ver)
        else:
            return ver

    def __init__(self, scalar_op, sync=True, inplace_pattern=None):
        if inplace_pattern is None:
            inplace_pattern = {}
        try:
            code = scalar_op.c_support_code_apply(None, "nodename")
            if code:
                raise SupportCodeError(scalar_op)
        except gof.utils.MethodNotDefined:
            pass
        self.scalar_op = scalar_op
        self.sync = sync
        self.inplace_pattern = inplace_pattern

    def c_src_kernel(self, node, nodename, nd):
        sio = StringIO()
        # print 'C_SRC_KERNEL', sio.getvalue()
        print("// %s" % str(node.op), file=sio)
        print("// node.op.destroy_map=%s" % str(
            getattr(node.op, 'destroy_map', None)), file=sio)
        for ipos, i in enumerate(node.inputs):
            print("//    Input  ", ipos, str(i.type), file=sio)
        for ipos, i in enumerate(node.outputs):
            print("//    Output ", ipos, str(i.type), file=sio)
        print("static __global__ void kernel_%s_%s_%s(unsigned int numEls" % (
            self.scalar_op.__class__.__name__, nodename, nd), file=sio)
        if (nd):
            print("\t,", ", ".join("const int dim%i" % i
                                           for i in xrange(nd)), file=sio)
        # declare inputs
        for ipos, i in enumerate(node.inputs):
            s = ", ".join(["const float * i%i_data" % ipos] +
                          ["int i%i_str_%i" % (ipos, d) for d in xrange(nd)])
            print("\t,", s, file=sio)
        # declare outputs
        for ipos, i in enumerate(node.outputs):
            s = ", ".join(["float * o%i_data" % ipos] +
                          ["int o%i_str_%i" % (ipos, d) for d in xrange(nd)])
            print("\t,", s, file=sio)
            #print >> sio, "\t,", ", ".join("int o%i_str_%i" % (ipos, d) for d in xrange(nd))
            #print >> sio, "\t,", "float * o%i_data" % ipos
        print("\t)\n{", file=sio)
        print("    const int idx = blockIdx.x * blockDim.x + threadIdx.x;", file=sio)
        print("    const int numThreads = blockDim.x * gridDim.x;", file=sio)

        # For each input that is a scalar which has been broadcasted to a tensor,
        #     load it into a local variable
        for ipos, i in enumerate(node.inputs):
            if _logical_scalar(i):
                print("    const float ii_i%i_value = i%i_data[0];" % (ipos, ipos), file=sio)

        # loop over the elements to be treated by this kernel call
        print("    for (int i = idx; i < numEls; i += numThreads) {", file=sio)
        # calculate the data pointers for all arguments
        print("        int ii = i;", file=sio)
        for ipos, i in enumerate(node.inputs):
            if not _logical_scalar(i):
                print("        const float * ii_i%i_data = i%i_data;" % (ipos, ipos), file=sio)
        for ipos, i in enumerate(node.outputs):
            print("        float * ii_o%i_data = o%i_data;" % (ipos, ipos), file=sio)
        for d in xrange(nd-1, -1, -1):
            if d > 0:
                print("        int pos%i = ii %% dim%i;" % (d, d), file=sio)
                print("        ii = ii / dim%i;" % d, file=sio)
            else:
                print("        int pos%i = ii;" % d, file=sio)

            for ipos, i in enumerate(node.inputs):
                if not _logical_scalar(i):
                    print("        ii_i%i_data += pos%i * i%i_str_%i;" % (ipos, d, ipos, d), file=sio)
            for ipos, i in enumerate(node.outputs):
                print("        ii_o%i_data += pos%i * o%i_str_%i;" % (ipos, d, ipos, d), file=sio)

        # perform the scalar operation on the input and output references
        # TODO: What if the scalar_op needs support_code??
        for ipos, i in enumerate(node.outputs):
            print("npy_%s o%d_i;" % (i.dtype, ipos), file=sio)
        task_code = self.scalar_op.c_code(
            Apply(self.scalar_op,
                  [scalar.Scalar(dtype=input.type.dtype).make_variable()
                   for input in node.inputs],
                  [scalar.Scalar(dtype=output.type.dtype).make_variable()
                   for output in node.outputs]),
            nodename + '_scalar_',
            get_str_list_logical_scalar(node),
            ['o%i_i' % ipos for ipos, i in enumerate(node.outputs)],
            sub=dict(fail='return;'))  # TODO: set a failure code somehow!!!
        print("       ", task_code, file=sio)
        for ipos, _ in enumerate(node.outputs):
            print("ii_o%i_data[0] = o%i_i;" % (ipos, ipos), file=sio)
        print("    }", file=sio)

        #indent = " "*(4*d+7)
        # for ipos, i in enumerate(node.inputs):
            #print >> sio, indent, "const float * i%i" % ipos, '= i%i_data', ''
        print("}", file=sio)

        # print sio.getvalue()
        return sio.getvalue()

    def c_src_kernel_tiling(self, node, nodename):
        """
        The kernel applies to problems with <= 5 dimensions.

        """
        # The kernel is intended to be structured roughly like this:
        """
        static __global__ void kernel()
        {
            for (int v = blockIdx.y; v < dim0; v += gridDim.x)
            {
                for (int w = blockIdx.y; w < dim1; w += gridDim.y)
                {
                    for (int x = threadIdx.x; x < dim2; x += blockDim.x)
                    {
                        for (int y = threadIdx.y; y < dim3; y += blockDim.y)
                        {
                            for (int z = threadIdx.z; z < dim4; z += blockDim.z)
                            {
                                out[v * out_stride[0] + ...] = f(in1[...],  in2[...])
                            }
                        }
                    }
                }
            }
        }

        """

        nd = node.outputs[0].type.ndim
        sio = StringIO()
        # print 'C_SRC_KERNEL', sio.getvalue()

        if nd in (4,):
            # print some leading comments to make the code easier to read
            print("// %s" % str(node.op), file=sio)
            print("// node.op.destroy_map=%s" % str(
                getattr(node.op, 'destroy_map', None)), file=sio)
            for ipos, i in enumerate(node.inputs):
                print("//    Input  ", ipos, str(i.type), file=sio)
            for ipos, i in enumerate(node.outputs):
                print("//    Output ", ipos, str(i.type), file=sio)
            print("static __global__ void kernel_%s_%s_%s(unsigned int numEls" % (
                    self.scalar_op.__class__.__name__,
                    nodename,
                    'tiling%i'%nd), file=sio)
            if (nd):
                print("\t,", ", ".join("const int dim%i" % i for i in xrange(nd)), file=sio)
            # declare inputs
            for ipos, i in enumerate(node.inputs):
                s = ", ".join(["const float * i%i_data" % ipos] + list("int i%i_str_%i" % (ipos, d) for d in xrange(nd)))
                print("\t,", s, file=sio)
            # declare outputs
            for ipos, i in enumerate(node.outputs):
                s = ", ".join(["float * o%i_data" % ipos] + list("int o%i_str_%i" % (ipos, d) for d in xrange(nd)))
                print("\t,", s, file=sio)
                #print >> sio, "\t,", ", ".join("int o%i_str_%i" % (ipos, d) for d in xrange(nd))
                #print >> sio, "\t,", "float * o%i_data" % ipos
            print("\t)\n{", file=sio)

            # For each input that is a scalar which has been broadcasted to a tensor,
            #     load it into a local variable
            print("    __shared__ float value0[%i];" % len(node.inputs), file=sio)
            print("    __shared__ int shared_dims[%(nd)s];" % locals(), file=sio)
            #print >> sio, "    __shared__ int shared_i_str[%(n_in)s][%(nd)s]"
            print("    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {", file=sio)
            for ipos, i in enumerate(node.inputs):
                if _logical_scalar(i):
                    print("    value0[%i] = i%i_data[0];" % (ipos, ipos), file=sio)
            for ipos in xrange(nd):
                print("    shared_dims[%i] = dim%i;" % (ipos, ipos), file=sio)
            print("    }", file=sio)
            print("    __syncthreads();", file=sio)

            if (nd == 4):
                print("""
                for (int pos0 = blockIdx.x; pos0 < shared_dims[0]; pos0 += gridDim.x)
                {
                    for (int pos1 = blockIdx.y; pos1 < shared_dims[1]; pos1 += gridDim.y)
                    {
                        //for (int pos2 = threadIdx.x; pos2 < shared_dims[2]; pos2 += blockDim.x)
                        for (int pos2 = threadIdx.y; pos2 < shared_dims[2]; pos2 += blockDim.y)
                        {
                            //for (int pos3 = threadIdx.y; pos3 < shared_dims[3]; pos3 += blockDim.y)
                            for (int pos3 = threadIdx.x; pos3 < shared_dims[3]; pos3 += blockDim.x)
                            {
                """, file=sio)
            else:
                raise NotImplementedError()

            for ipos, i in enumerate(node.inputs):
                if not _logical_scalar(i):
                    print("        const float * ii_i%i_data = i%i_data;" % (ipos, ipos), file=sio)
            for ipos, i in enumerate(node.outputs):
                print("        float * ii_o%i_data = o%i_data;" % (ipos, ipos), file=sio)
            for d in xrange(nd):
                for ipos, i in enumerate(node.inputs):
                    if not _logical_scalar(i):
                        print("        ii_i%i_data += pos%i * i%i_str_%i;" % (ipos, d, ipos, d), file=sio)
                for ipos, i in enumerate(node.outputs):
                    print("        ii_o%i_data += pos%i * o%i_str_%i;" % (ipos, d, ipos, d), file=sio)

            # perform the scalar operation on the input and output references
            # TODO: What if the scalar_op needs support_code??
            task_code = self.scalar_op.c_code(
                    Apply(self.scalar_op,
                        [scalar.Scalar(dtype=input.type.dtype).make_variable()
                         for input in node.inputs],
                        [scalar.Scalar(dtype=output.type.dtype).make_variable()
                         for output in node.outputs])
                    , nodename + '_scalar_'
                    , get_str_list_logical_scalar(node, value_str='value0[%i]')
                    , ['ii_o%i_data[0]'%ipos for ipos, i in enumerate(node.outputs)]
                    , sub=dict(fail='return;'))  # TODO: set a failure code somehow!!!
            print("       ", task_code, file=sio)

            print("    }" * nd, file=sio)

            # TODO: insert runtime stride checks that select the best loop order either here, or in
            # the host code that launched the  kernel (host code probably better spot)

            #indent = " "*(4*d+7)
            # for ipos, i in enumerate(node.inputs):
                #print >> sio, indent, "const float * i%i" % ipos, '= i%i_data', ''
            print("}", file=sio)

        print(sio.getvalue())
        return sio.getvalue()

    def c_src_kernel_tiling_less_registers(self, node, nodename):
        """
        The kernel applies to problems with <= 5 dimensions.

        """
        nd = node.outputs[0].type.ndim
        n_in = len(node.inputs)
        n_out = len(node.outputs)
        sio = StringIO()

        if nd not in (2,):
            return sio.getvalue()

        # print some leading comments to make the code easier to read
        print("// %s" % str(node.op), file=sio)
        print("// node.op.destroy_map=%s" % str(
            getattr(node.op, 'destroy_map', None)), file=sio)
        for ipos, i in enumerate(node.inputs):
            print("//    Input  ", ipos, str(i.type), file=sio)
        for ipos, i in enumerate(node.outputs):
            print("//    Output ", ipos, str(i.type), file=sio)
        print("static __global__ void kernel_%s_%s_%s(unsigned int numEls" % (
                self.scalar_op.__class__.__name__,
                nodename,
                'tiling%i_less_registers'%nd), file=sio)
        if (nd):
            print("\t,", ", ".join("const int dim%i" % i for i in xrange(nd)), file=sio)
        # declare inputs
        for ipos, i in enumerate(node.inputs):
            s = ", ".join(["const float * i%i_data_0" % ipos] + list("int i%i_str_%i" % (ipos, d) for d in xrange(nd)))
            print("\t,", s, file=sio)
        # declare outputs
        for ipos, i in enumerate(node.outputs):
            s = ", ".join(["float * o%i_data_0" % ipos] + list("int o%i_str_%i" % (ipos, d) for d in xrange(nd)))
            print("\t,", s, file=sio)
            #print >> sio, "\t,", ", ".join("int o%i_str_%i" % (ipos, d) for d in xrange(nd))
            #print >> sio, "\t,", "float * o%i_data" % ipos
        print("\t)\n{", file=sio)

        # TODO: Setting these to true makes the function fail SOMETIMES.  I don't know why yet.
        use_shared_stride = False
        use_shared_limits = False

        def decl_limits(nd):
            if use_shared_limits:
                print("__shared__ float * limits[%(nd)s];" % locals(), file=sio)

        def stride(io, p, d):
            if use_shared_stride:
                return "s%s_str[%i][%i]" % (io, p, d)
            else:
                return "%s%i_str_%i" % (io, p, d)
        def limits(d):
            if use_shared_limits:
                return "limits[%i]" % d
            else:
                return "limits%i" % d

        def decl_shared_stride(nin, nout, nd):
            if not use_shared_stride:
                return
            print("""
            __shared__ int si_str[%(nin)s][%(nd)s];
            __shared__ int so_str[%(nout)s][%(nd)s];
            if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
            """ % locals(), file=sio)
            for i in xrange(nin):
                for d in xrange(nd):
                    print("si_str[%(i)s][%(d)s] = i%(i)s_str_%(d)s;" % locals(), file=sio)
            for i in xrange(n_out):
                for d in xrange(nd):
                    print("so_str[%(i)s][%(d)s] = o%(i)s_str_%(d)s;" % locals(), file=sio)
            print("} __syncthreads();", file=sio)

        def calc_limit(d):
            s = stride('o', 0, d)
            lname = limits(d)
            if use_shared_limits:
                print("if ((threadIdx.x == 0) && (threadIdx.y == 0)) {", file=sio)
                if d == 0:
                    print("%(lname)s = o0_data_0 + dim%(d)s * %(s)s;" % locals(), file=sio)
                else:
                    dm1 = d - 1
                    print("%(lname)s = o0_data_%(dm1)s + dim%(d)s * %(s)s;" % locals(), file=sio)
                print("} __syncthreads();", file=sio)
            else:
                if d == 0:
                    print("const float * %(lname)s = o0_data_0 + dim%(d)s * %(s)s;" % locals(), file=sio)
                else:
                    dm1 = d - 1
                    print("const float * %(lname)s = o0_data_%(dm1)s + dim%(d)s * %(s)s;" % locals(), file=sio)

        def decl_ptrs(d, offset):
            dm1 = d - 1
            assert dm1 >= 0
            for i in xrange(n_in):
                s = stride('i', i, d)
                print("const float * i%(i)s_data_%(d)s = i%(i)s_data_%(dm1)s + %(offset)s * %(s)s;" % locals(), file=sio)
            for i in xrange(n_out):
                s = stride('o', i, d)
                print("float * o%(i)s_data_%(d)s = o%(i)s_data_%(dm1)s + %(offset)s * %(s)s;" % locals(), file=sio)

        def inc_ptrs(d, amt):
            for i in xrange(n_in):
                s = stride('i', i, d)
                print("i%(i)s_data_%(d)s += %(amt)s * %(s)s;" % locals(), file=sio)
            for i in xrange(n_out):
                s = stride('o', i, d)
                print("o%(i)s_data_%(d)s += %(amt)s * %(s)s;" % locals(), file=sio)

        def while_limit(d):
            lname = limits(d)
            print("while (o0_data_%(d)s < %(lname)s) { " % locals(), file=sio)

        def end_while(d):
            print("}", file=sio)

        def task_code(d):
            print(self.scalar_op.c_code(
                Apply(self.scalar_op,
                    [scalar.Scalar(dtype=input.type.dtype).make_variable()
                     for input in node.inputs],
                    [scalar.Scalar(dtype=output.type.dtype).make_variable()
                     for output in node.outputs])
                , nodename + '_scalar_'
                , ['i%i_data_%i[0]'%(ipos, d) for ipos, i in enumerate(node.inputs)]
                , ['o%i_data_%i[0]'%(ipos, d) for ipos, i in enumerate(node.outputs)]
                , sub=dict(fail='return;')), file=sio)  # TODO: set a failure code somehow!!!

        if nd == 4:
            decl_shared_stride(n_in, n_out, nd)
            decl_limits(nd)
            calc_limit(0)
            inc_ptrs(0, 'blockIdx.x')
            while_limit(0)
            if 1:
                calc_limit(1)
                decl_ptrs(1, 'blockIdx.y')
                while_limit(1)
                if 1:
                    calc_limit(2)
                    decl_ptrs(2, 'threadIdx.y')
                    while_limit(2)
                    if 1:
                        calc_limit(3)
                        decl_ptrs(3, 'threadIdx.x')
                        while_limit(3)
                        if 1:
                            task_code(3)
                            inc_ptrs(3, 'blockDim.x')
                        end_while(3)
                        inc_ptrs(2, 'blockDim.y')
                    end_while(2)
                    inc_ptrs(1, 'gridDim.y')
                end_while(1)
                inc_ptrs(0, 'gridDim.x')
            end_while(0)

        print("}", file=sio)
        print(sio.getvalue())
        return sio.getvalue()

    def c_src_kernel_Ccontiguous(self, node, nodename):
        sio = StringIO()
        # print 'C_SRC_KERNEL', sio.getvalue()

        print("// %s" % str(node.op), file=sio)
        print("// node.op.destroy_map=%s" % str(
            getattr(node.op, 'destroy_map', None)), file=sio)
        for ipos, i in enumerate(node.inputs):
            print("//    Input  ", ipos, str(i.type), file=sio)
        for ipos, i in enumerate(node.outputs):
            print("//    Output ", ipos, str(i.type), file=sio)
        print("static __global__ void kernel_%s_%s_Ccontiguous (unsigned int numEls" % (self.scalar_op.__class__.__name__, nodename), file=sio)
        # declare inputs
        for ipos, i in enumerate(node.inputs):
            print("\t,", "const float * i%i_data" % ipos, file=sio)
        # declare outputs
        for ipos, i in enumerate(node.outputs):
            print("\t,", "float * o%i_data" % ipos, file=sio)
        print("\t)\n{", file=sio)
        print("    const int idx = blockIdx.x * blockDim.x + threadIdx.x;", file=sio)
        print("    const int numThreads = blockDim.x * gridDim.x;", file=sio)

        # For each input that is a scalar which has been broadcasted to a tensor,
        #     load it into a local variable
        for ipos, i in enumerate(node.inputs):
            if _logical_scalar(i):
                print("    const float ii_i%i_value = i%i_data[0];" % (ipos, ipos), file=sio)

        # loop over the elements to be treated by this kernel call
        print("    for (int i = idx; i < numEls; i += numThreads) {", file=sio)
        # perform the scalar operation on the input and output references
        # TODO: What if the scalar_op needs support_code??
        for ipos, i in enumerate(node.outputs):
            print("npy_%s o%d_i;" % (i.dtype, ipos), file=sio)
        task_code = self.scalar_op.c_code(
                Apply(self.scalar_op,
                    [scalar.Scalar(dtype=input.type.dtype).make_variable()
                     for input in node.inputs],
                    [scalar.Scalar(dtype=output.type.dtype).make_variable()
                     for output in node.outputs])
                , nodename + '_scalar_'
                #, ['i%i_data[i]'%ipos for ipos, i in enumerate(node.inputs)]
                , get_str_list_logical_scalar(node, data_str='i%i_data[i]')
                , ['o%i_i'%ipos for ipos, i in enumerate(node.outputs)]
                , sub=dict(fail='return;'))  # TODO: set a failure code somehow!!!
        print("       ", task_code, file=sio)
        for ipos, _ in enumerate(node.outputs):
            print("o%i_data[i] = o%i_i;" % (ipos, ipos), file=sio)
        print("    }", file=sio)
        print("}", file=sio)

        # print sio.getvalue()
        return sio.getvalue()

    def c_src_callkernel(self, node, nodename):
        #
        # This function serves three main goals:
        #
        # The first is stride unpacking:
        # it accepts input and output arguments as
        #    float * , int*
        # pairs, and it constructs a kernel function call where inputs and arguments are named
        # like
        #    float *, int, int, int ...
        #
        # The second is to recognize when any dimensions can be collapsed as
        # being contiguous. That mean that we can merge that dimensions with another
        # one for all inputs/outputs and have the same retusuls (confusing... read code)
        #
        # The thrid is to make a special case for scalar element. We allow the collapsing of them.
        # In the ccontiguous and not contiguous case, we use registers to lower the number of memory access.

        # TODO: make a special case for broadcasting, to store the data in shared memory.

        nd = node.outputs[0].type.ndim
        nb_inputs = len(node.inputs)
        nb_outputs = len(node.outputs)
        d = dict()
        # input_params and output_params go into the function declaration/definition
        input_params = ", ".join("const float * i%i_data, const int * i%i_str"%(ipos, ipos)
                for ipos in xrange(len(node.inputs)))
        output_params = ", ".join("float * o%i_data, const int * o%i_str"%(ipos, ipos)
                for ipos in xrange(len(node.outputs)))

        # input_args and output_args go into the recursive call.
        input_args = ", ".join("i%i_data, i%i_str"%(ipos, ipos)
                for ipos in xrange(len(node.inputs)))
        output_args = ", ".join("o%i_data, o%i_str"%(ipos, ipos)
                for ipos in xrange(len(node.outputs)))

        prod_dims = '*'.join(["dims[%i]"%di for di in xrange(nd)]+['1'])

        scalar_op = self.scalar_op.__class__.__name__

        sio = StringIO()
        print("""
        static void can_collapse_%(nodename)s(int nd, const int * dims, const int * strides, int collapse[])
        {
            //can we collapse dims[i] and dims[i-1]
            for(int i=nd-1;i>0;i--){
                if(strides[i]*dims[i]==strides[i-1]){//the dims nd-1 are not strided again dimension nd
                    collapse[i]=1;
                }else collapse[i]=0;
            }
        }
        """ % locals(), file=sio)
        print("""
        static int callkernel_%(nodename)s(unsigned int numEls, const int d,
            const int * dims,
            %(input_params)s,
            %(output_params)s)
        {
            numEls = %(prod_dims)s;
        """ % locals(), file=sio)
        if self.verbose:
            print("""
                std::cerr << "calling kernel_%(scalar_op)s_%(nodename)s     w numEls" << numEls << " dims"<< d << "\\n";
            """ % locals(), file=sio)
            print('std::cerr << ' + " << ' ' <<  ".join(['"  "']+list("dims[%i]"%di
                for di in xrange(nd)) + ["'\\n';"]), file=sio)
        if self.verbose > 1:
            for ipos in xrange(len(node.inputs)):
                print("""
                std::cerr << "   %(ipos)s data strides" <<
                """ % locals() + " << ' ' <<  ".join(["i%s_data"%ipos]
                + list("i%s_str[%i]"%(ipos, di) for di in xrange(nd))) + ''' << "\\n"; ''', file=sio)

            for ipos in xrange(len(node.outputs)):
                print("""
                std::cerr << "   %(ipos)s data strides" <<
                """ % locals() + " << ' ' <<  ".join(["o%s_data"%ipos]
                    + list("o%s_str[%i]"%(ipos, di) for di in xrange(nd))) + ''' << "\\n"; ''', file=sio)
    # collapse dimension that are broadcast in all inputs.
    # need to be done before contiguous collapse as it will break it.
    # do the dimensions and the strides
        if nd > 0:
            print("int local_dims[%(nd)s];" % locals(), file=sio)
        else:
            print("int *local_dims=NULL;", file=sio)
        if nb_inputs > 0 and nd > 0:
            print("""
            int local_str[%(nb_inputs)s][%(nd)s];
            int local_ostr[%(nb_outputs)s][%(nd)s];
            """ % locals(), file=sio)
        else:
            print("""
            int local_str[1][1];
            int local_ostr[1][1];
            """, file=sio)
        print("""
        int nd_collapse = %(nd)s;
        for(int i=0;i<%(nd)s;i++){//init new dim
          local_dims[i]=dims[i];
        }
        """ % locals(), file=sio)
        for ipos in xrange(len(node.inputs)):
            print("""
            for(int i=0;i<%(nd)s;i++){//init new strides
              local_str[%(ipos)s][i]=i%(ipos)s_str[i];
            }
            """ % locals(), file=sio)
        for ipos in xrange(len(node.outputs)):
            print("""
            for(int i=0;i<%(nd)s;i++){//init new strides
              local_ostr[%(ipos)s][i]=o%(ipos)s_str[i];
            }
            """ % locals(), file=sio)
        if self.verbose > 2:
            print('std::cerr <<"before broadcast collapse\\n";', file=sio)
            print('std::cerr<< "nd_collapse "<< nd_collapse << "\\n"; ', file=sio)
            print('std::cerr << "local_dims";', file=sio)
            for d in xrange(nd):
                print('std::cerr << " " << local_dims[%(d)s]; ' % locals(), file=sio)
            print('std::cerr << "\\n";', file=sio)
            if nd > 0:
                for ipos in xrange(len(node.inputs)):
                    print('std::cerr << " local_str inputs %(ipos)s: " <<'%locals() + \
                        ' << " " << '.join(["local_str[%s][%s]" % (ipos, x) for x in xrange(nd)])+'<<"\\n";', file=sio)
                    for ipos in xrange(len(node.outputs)):
                        print('std::cerr << " local_ostr inputs %(ipos)s: " <<'%locals() + \
                        ' << " " << '.join(["local_ostr[%s][%s]" % (ipos, x) for x in xrange(nd)])+'<<"\\n";', file=sio)

        print("""
        for(int id=0;id<nd_collapse;id++){

          bool all_broadcast=true;
          for(int input_id=0;input_id<%(nb_inputs)s;input_id++){
            if(local_str[input_id][id]!=0 || local_dims[id]!=1) all_broadcast= false;
          }
          for(int input_id=0;input_id<%(nb_outputs)s;input_id++){
            if(local_ostr[input_id][id]!=0 || local_dims[id]!=1) all_broadcast= false;
          }
          if(all_broadcast){
            for(int j=id+1;j<nd_collapse;j++)//remove dims i from the array
              local_dims[j-1]=local_dims[j];
            for(int input_id=0;input_id<%(nb_inputs)s;input_id++){
              for(int j=id+1;j<nd_collapse;j++){//remove dims i from the array
                local_str[input_id][j-1]=local_str[input_id][j];
              }
            }
            for(int output_id=0;output_id<%(nb_outputs)s;output_id++){
              for(int j=id+1;j<nd_collapse;j++){//remove dims i from the array
                local_ostr[output_id][j-1]=local_ostr[output_id][j];
              }
            }
            nd_collapse--; id--;
          }
        }
        """%locals(), file=sio)

        if self.verbose > 2:
            print('std::cerr <<"after broadcast collapse\\n";', file=sio)
            print('std::cerr<< "nd_collapse "<< nd_collapse << "\\n"; ', file=sio)
            print('std::cerr << "local_dims";', file=sio)
            for d in xrange(nd):
                print('std::cerr << " " << local_dims[%(d)s]; '%locals(), file=sio)
            print('std::cerr << "\\n";', file=sio)
            if nd > 0:
                for ipos in xrange(len(node.inputs)):
                    print('std::cerr << " local_str %(ipos)s: " <<'%locals()+' << " " << '.join(["local_str[%s][%s]" % (ipos, x) for x in xrange(nd)])+'<<"\\n";', file=sio)
                    for ipos in xrange(len(node.outputs)):
                        print('std::cerr << " local_ostr %(ipos)s: " <<'%locals()+' << " " << '.join(["local_ostr[%s][%s]" % (ipos, x) for x in xrange(nd)])+'<<"\\n";', file=sio)
    # collapse contiguous dimensions (ignoring scalars, generic version(collapse any dimensions, right, left, middle))
    # this is a good idea because we make less index calculation in the gpu.

        if nd > 0:
            print("int nd_collapse_[%(nd)s] = {"%locals() + ','.join(['1' for x in xrange(nd)]) + "};", file=sio)
        else:
            print("int *nd_collapse_ = NULL;", file=sio)
        for ipos in xrange(len(node.inputs)):
            if not _logical_scalar(node.inputs[ipos]):
                if nd > 0:
                    print("""
                        int nd_collapse_%(ipos)s[%(nd)s] = {"""%locals() + ','.join(['1' for x in xrange(nd)]) + "};", file=sio)
                else:
                    print("""
                        int *nd_collapse_%(ipos)s = NULL;"""%locals(), file=sio)
                print("""
can_collapse_%(nodename)s(nd_collapse, local_dims, local_str[%(ipos)s], nd_collapse_%(ipos)s);
for(int i=0;i<nd_collapse;i++){
if(nd_collapse_%(ipos)s[i]==0)
nd_collapse_[i]=0;
}
                """ % locals(), file=sio)
                if self.verbose > 1:
                    print("""
                    std::cerr<< "nd_collapse_%(ipos)s "<<
                    """%locals(), file=sio)
                    print(' << " " << '.join(["nd_collapse_%s[" % ipos + str(i)+"]" for i in xrange(nd)]), file=sio)
                    print('<< "\\n";', file=sio)

    # update the local stride.
        for ipos in xrange(len(node.inputs)):
            print("""
            for(int i=nd_collapse-1;i>0;i--){
              if(nd_collapse_[i]==1){
                local_str[%(ipos)s][i-1]=local_str[%(ipos)s][i];//set new strides
                for(int j=i+1;j<nd_collapse;j++)//remove stride i from the array
                  local_str[%(ipos)s][j-1]=local_str[%(ipos)s][j];
                }
            }
            """%locals(), file=sio)

        for ipos in xrange(len(node.outputs)):
            print("""
            for(int i=nd_collapse-1;i>0;i--){
              if(nd_collapse_[i]==1){
                local_ostr[%(ipos)s][i-1]=local_ostr[%(ipos)s][i];//set new strides
                for(int j=i+1;j<nd_collapse;j++)//remove stride i from the array
                  local_ostr[%(ipos)s][j-1]=local_ostr[%(ipos)s][j];
                }
            }
            """%locals(), file=sio)

    # update the local dims.
        print("""
        for(int i=nd_collapse-1;i>0;i--){
          if(nd_collapse_[i]==1){
            local_dims[i-1]*=local_dims[i];//set new dims
            for(int j=i+1;j<nd_collapse;j++)//remove dims i from the array
              local_dims[j-1]=local_dims[j];
          }
        }
        """%locals(), file=sio)

    # update the new number of dim
        print("""
        for(int i=1, end=nd_collapse;i<end;i++){
          if(nd_collapse_[i]==1)nd_collapse--;
        }
        if(nd_collapse == 1 """%locals(), file=sio)
        l = ["local_str[%s][nd_collapse-1]==1 "%ipos for ipos in xrange(len(node.inputs)) if not _logical_scalar(node.inputs[ipos])]
        l += ["local_ostr[%s][nd_collapse-1]==1 "%ipos for ipos in xrange(len(node.outputs)) if not _logical_scalar(node.outputs[ipos])]
        if len(l) > 0:
            print(" && ", " && ".join(l), file=sio)
        print("""){nd_collapse=0;} """, file=sio)

        if self.verbose:
            print('std::cerr <<"after can_collapse\\n";', file=sio)
            print("""std::cerr << "nd_collapse " << nd_collapse << "\\n"; """ % locals(), file=sio)
        if self.verbose > 1:
            for d in xrange(nd):
                print('std::cerr << " " << local_dims[%(d)s]; '%locals(), file=sio)
            print('std::cerr << "\\n";', file=sio)
            if nd > 0:
                for ipos in xrange(len(node.inputs)):
                    print('std::cerr << " local_str %(ipos)s: " <<'%locals()+' << " " << '.join(["local_str[%s][%s]"%(ipos, x) for x in xrange(nd)])+'<<"\\n";', file=sio)
                    for ipos in xrange(len(node.outputs)):
                        print('std::cerr << " local_ostr %(ipos)s: " <<'%locals()+' << " " << '.join(["local_ostr[%s][%s]"%(ipos, x) for x in xrange(nd)])+'<<"\\n";', file=sio)

        def launch_Ccontiguous(nodename, scalar_op, sync=True):
            kernel_call_args = ["numEls"]
            for ipos in xrange(len(node.inputs)):
                kernel_call_args.append("i%i_data"%ipos)
            for ipos in xrange(len(node.outputs)):
                kernel_call_args.append("o%i_data"%ipos)
            kernel_call_args = ", ".join(kernel_call_args)
            verb = ""
            if self.verbose:
                verb = 'std::cerr << "   Running ccontiguous version\\n";'
            print("""
                //first use at least a full warp
                int threads_per_block = std::min(numEls,  (unsigned int)32); //WARP SIZE

                //next start adding multiprocessors
                int n_blocks = std::min(numEls/threads_per_block + (numEls %% threads_per_block?1:0), (unsigned int)30); // UP TO NUMBER OF MULTIPROCESSORS

                // next start adding more warps per multiprocessor
                if (threads_per_block * n_blocks < numEls)
                    threads_per_block = std::min(numEls/n_blocks, (unsigned int)NUM_VECTOR_OP_THREADS_PER_BLOCK);
                kernel_%(scalar_op)s_%(nodename)s_Ccontiguous<<<n_blocks, threads_per_block>>>(%(kernel_call_args)s);

                //std::cerr << "calling callkernel returned\\n";
                """ % locals(), file=sio)
            if sync:
                print("""
                CNDA_THREAD_SYNC;
                cudaError_t err = cudaGetLastError();
                if( cudaSuccess != err)
                {
                    PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s: %%s.\\n    n_blocks=%%i threads_per_block=%%i\\n   Call: %%s\\n",
                         "GpuElemwise %(nodename)s %(scalar_op)s", cudaGetErrorString(err),
                         n_blocks, threads_per_block,
                         "kernel_%(scalar_op)s_%(nodename)s_Ccontiguous<<<n_blocks, threads_per_block>>>(%(kernel_call_args)s)");
                    return -1;

                }
                %(verb)s
                return 0;
                """ % locals(), file=sio)
            else:
                print(" return 0; " % locals(), file=sio)

        def launch_General(nodename, scalar_op, force_nd, sync=True):
            # kernel_call_args are used to invoke the cuda kernel
            local = "local_"
            kernel_call_args = ["numEls"]
            kernel_call_args.extend(local+"dims[%i]"%di for di in xrange(force_nd))
            for ipos in xrange(len(node.inputs)):
                kernel_call_args += ["i%i_data"%ipos] + list(local+"str[%i][%i]"%(ipos, di) for di in xrange(force_nd))
                #strides = ", ".join("i%i_str[%i]"%(ipos, di) for di in xrange(force_nd))
                #kernel_call_args.append( "%s, i%i_data" % (strides, ipos))
            for ipos in xrange(len(node.outputs)):
                kernel_call_args += ["o%i_data"%ipos] + list(local+"ostr[%i][%i]"%(ipos, di) for di in xrange(force_nd))
                #strides = ", ".join("o%i_str[%i]"%(ipos, di) for di in xrange(force_nd))
                #kernel_call_args.append( "%s, o%i_data" % (strides, ipos))
            if self.verbose:
                print("""
                    std::cerr << "   Running general version with %(force_nd)s  dims\\n";
                    """%locals(), file=sio)
                print("std::cerr << " + ' << " " << '.join(kernel_call_args)+' << "\\n";', file=sio)
                # std::cerr << numEls << dims[0] << i0_data, i0_str[0] << o0_data, o0_str[0]\n;

            kernel_call_args = ", ".join(kernel_call_args)

            print("""
                //first use at least a full warp
                int threads_per_block = std::min(numEls, (unsigned int)32); //WARP SIZE

                //next start adding multiprocessors
                int n_blocks = std::min(numEls/threads_per_block + (numEls %% threads_per_block?1:0), (unsigned int)30); // UP TO NUMBER OF MULTIPROCESSORS

                // next start adding more warps per multiprocessor
                if (threads_per_block * n_blocks < numEls)
                    threads_per_block = std::min(numEls/n_blocks, (unsigned int)NUM_VECTOR_OP_THREADS_PER_BLOCK);

                kernel_%(scalar_op)s_%(nodename)s_%(force_nd)s<<<n_blocks, threads_per_block>>>(%(kernel_call_args)s);
                """ % locals(), file=sio)
            if sync:
                print("""
                CNDA_THREAD_SYNC;
                cudaError_t err = cudaGetLastError();
                if( cudaSuccess != err)
                {
                    PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s: %%s.\\n    n_blocks=%%i threads_per_block=%%i\\n   Call: %%s\\n",
                         "GpuElemwise %(nodename)s %(scalar_op)s", cudaGetErrorString(err),
                         n_blocks, threads_per_block,
                         "kernel_%(scalar_op)s_%(nodename)s_Ccontiguous<<<n_blocks, threads_per_block>>>(%(kernel_call_args)s)");
                    return -1;

                }
                return 0;
                """ % locals(), file=sio)
            else:
                print(" return 0; " % locals(), file=sio)
        print("if(numEls==0) return 0;", file=sio)
        print("switch (nd_collapse==0?0:min(%(nd)s,nd_collapse)) {"%locals(), file=sio)
        print("case 0: {", file=sio)
        launch_Ccontiguous(nodename, scalar_op, self.sync)
        print("        } break;", file=sio)
        for i in xrange(1, nd+1):
            print("case "+str(i)+": {", file=sio)
            launch_General(nodename, scalar_op, i, self.sync)
            print("        } break;", file=sio)

        print("}", file=sio)  # end case
        print("return -2;", file=sio)  # should not get to this point
        print("}", file=sio)  # end fct

        # N.B. cudaGetLastError is called by c_code
        return sio.getvalue()

    def c_support_code_apply(self, node, nodename):
        nd = node.outputs[0].type.ndim
        defines = """
#define INTDIV_POW2(a, b) (a >> b)
#define INTMOD_POW2(a, b) (a & ((1<<b)-1))
        """
        kernels = "".join(
            [self.c_src_kernel(node, nodename, x) for x in xrange(1, nd + 1)]
            + [self.c_src_kernel_Ccontiguous(node, nodename)]
            + [self.c_src_callkernel(node, nodename)])
        return defines + kernels

    def c_support_code(self):
        return self.scalar_op.c_support_code()

    def c_code(self, node, nodename, inputs, outputs, sub):
        d = dict(sub)
        nd = node.outputs[0].type.ndim
        d.update(locals())
        sio = StringIO()
        nin = len(inputs)
        nout = len(outputs)
        fail = sub['fail']
        opname = str(self.scalar_op)
        initial_dims = ','.join('1' for i in xrange(nd))
        if 1 or self.scalar_op == scalar.pow:
            print("""
        //std::cerr << "C_CODE %(opname)s START\\n";
        //standard elemwise size checks
            """ % locals(), file=sio)
        if nd > 0:
            print("""
            int dims[%(nd)s] = {%(initial_dims)s};
            """ % locals(), file=sio)
        else:
            print("""
            int *dims = NULL;
            """, file=sio)

        # check that all inputs have valid dimensions
        emitted_inames = {}
        for id, iname in enumerate(inputs):
            if iname in emitted_inames:
                assert emitted_inames[iname] is node.inputs[id]
                continue

            # with python 2.4 (at least), if a broadcastable pattern is made of
            # numpy.bool_ instead of bool, calling int() once is not enough.
            broadcasts = map(int, map(int, node.inputs[id].broadcastable))
            broadcasts = ', '.join(map(str, broadcasts))
            nd = node.inputs[id].ndim
            if nd > 0:
                print("""
                int broadcasts_%(iname)s[%(nd)s] = {%(broadcasts)s};
                """ % locals(), file=sio)
            else:
                print("""
                int *broadcasts_%(iname)s = NULL;
                """ % locals(), file=sio)
            emitted_inames[iname] = node.inputs[id]

        # check that all inputs have valid dimensions
        emitted_inames = {}
        for id, iname in enumerate(inputs):
            if iname in emitted_inames:
                continue
            print("""
        //std::cerr << "C_CODE %(opname)s checking input %(iname)s\\n";
        if (%(nd)s != %(iname)s->nd)
        {
            PyErr_Format(PyExc_TypeError,
                         "need %(nd)s dims, not %%i", %(iname)s->nd);
            %(fail)s;
        }
        for (int i = 0; i< %(nd)s; ++i)
        {
            dims[i] = (dims[i] == 1) ? CudaNdarray_HOST_DIMS(%(iname)s)[i] : dims[i];
            if ((!(broadcasts_%(iname)s[i] &&
                 CudaNdarray_HOST_DIMS(%(iname)s)[i] == 1)) &&
                (dims[i] != CudaNdarray_HOST_DIMS(%(iname)s)[i]))
            {
                //std::cerr << "C_CODE %(opname)s checking input %(iname)s failed\\n";
                PyErr_Format(PyExc_ValueError,
                             "GpuElemwise. Input dimension mis-match. Input"
                             " %(id)d (indices start at 0) has shape[%%i] == %%i"
                             ", but the output's size on that axis is %%i.",
                             i,
                             CudaNdarray_HOST_DIMS(%(iname)s)[i],
                             dims[i]
                            );
                %(fail)s;
            }
        }
            """ % locals(), file=sio)
            emitted_inames[iname] = True

        # check that all outputs have valid dimensions
        for idx, oname in enumerate(outputs):
            if idx not in self.inplace_pattern.keys():
                print("""
        for (int i = 0; (i< %(nd)s) && (%(oname)s); ++i) {
            if (dims[i] != CudaNdarray_HOST_DIMS(%(oname)s)[i])
            {
                Py_DECREF(%(oname)s);
                %(oname)s = NULL;
            }
        }
        if (%(oname)s && !CudaNdarray_is_c_contiguous(%(oname)s))
        {
            Py_XDECREF(%(oname)s);
            %(oname)s = NULL;
        }
        if (NULL == %(oname)s)
        {
            %(oname)s = (CudaNdarray*)CudaNdarray_New();
            if (!%(oname)s)
            {
                //error string already set
                %(fail)s;
            }
            if (CudaNdarray_alloc_contiguous(%(oname)s, %(nd)s, dims))
            {
                //error string already set
                Py_DECREF(%(oname)s);
                %(oname)s = NULL;
                %(fail)s;
            }
        }
        //std::cerr << "ELEMWISE NEW %(oname)s nd" << %(oname)s->nd << "\\n";
        //std::cerr << "ELEMWISE NEW %(oname)s data" << %(oname)s->devdata << "\\n";
        """ % locals(), file=sio)
            else:
                input_idx = self.inplace_pattern[idx]
                iname = inputs[input_idx]
                print("""
        Py_XDECREF(%(oname)s);
        %(oname)s = %(iname)s;
        Py_INCREF(%(oname)s);
        for (int i = 0; (i< %(nd)s) && (%(oname)s); ++i) {
            if (dims[i] != CudaNdarray_HOST_DIMS(%(oname)s)[i])
            {
                PyErr_Format(PyExc_ValueError,
                             "GpuElemwise. Output dimension mis-match. Output"
                             " %(idx)d (indices start at 0), working inplace"
                             " on input %(input_idx)s, has shape[%%i] == %%i"
                             ", but the output's size on that axis is %%i.",
                             i,
                             CudaNdarray_HOST_DIMS(%(oname)s)[i],
                             dims[i]
                            );
                Py_DECREF(%(oname)s);
                %(oname)s = NULL;
                %(fail)s;
            }
        }
        //std::cerr << "ELEMWISE NEW %(oname)s nd" << %(oname)s->nd << "\\n";
        //std::cerr << "ELEMWISE NEW %(oname)s data" << %(oname)s->devdata << "\\n";
        """ % locals(), file=sio)

        print("""
        {
            //new block so that failure gotos don't skip over variable initialization
            //std::cerr << "calling callkernel\\n";
            if (callkernel_%(nodename)s(1, 0, dims
            """ % locals(), file=sio)
        for iname in inputs:
            print("""
                        , CudaNdarray_DEV_DATA(%(iname)s), CudaNdarray_HOST_STRIDES(%(iname)s)
            """ % locals(), file=sio)
        for oname in outputs:
            print("""
                        , CudaNdarray_DEV_DATA(%(oname)s), CudaNdarray_HOST_STRIDES(%(oname)s)
            """ % locals(), file=sio)
        print("""
                        ))
            {
                 // error
            """, file=sio)
        for oname in outputs:
            print("""
                Py_DECREF(%(oname)s);
                %(oname)s = NULL;
                """ % locals(), file=sio)
        print("""
                %(fail)s;
            }
            else // no error
            {
            }
        }
        //std::cerr << "C_CODE %(opname)s END\\n";
        """ % locals(), file=sio)
        # print sio.getvalue()
        return sio.getvalue()


class ErfinvGPU(Erfinv):
    """
    Provides a c-code implementation of the inverse error function for GPU.

    Notes
    -----
    We do not add this c_code to theano.scalar.basic_scipy.Erfinv, as we
    currently rely on Nvidia's cublas library to provide the erfinv
    c-implementation (which requires different c_headers). As it stands,
    theano.scalar.basic_scipy.Erfinv does not have c_code as scipy does not
    export the required C function.

    """

    def c_headers(self):
        return ['math_functions.h', 'cublas_v2.h']

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = erfinv(%(x)s);" % locals()

erfinv_gpu = ErfinvGPU(upgrade_to_float_no_complex, name='erfinv_gpu')


class ErfcxGPU(Erfinv):
    """
    Provides a c-code implementation of the scaled complementary error function
    for GPU.

    Notes
    -----
    We do not add this c_code to theano.scalar.basic_scipy.Erfcx, as we
    currently rely on Nvidia's cublas library to provide the erfcx
    c-implementation (which requires different c_headers). As it stands,
    theano.scalar.basic_scipy.Erfcx does not have c_code as scipy does not
    export the required C function.

    """

    def c_headers(self):
        return ['math_functions.h', 'cublas_v2.h']

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = erfcx(%(x)s);" % locals()

erfcx_gpu = ErfcxGPU(upgrade_to_float_no_complex, name='erfcx_gpu')
